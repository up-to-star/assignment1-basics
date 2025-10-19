import os
import argparse
from contextlib import nullcontext
from pyexpat import model
import numpy as np
import torch
from tqdm import tqdm
from cs336_basics.dataset.utils import get_batch
from cs336_basics.layer.transformer import TransfromerLM
from cs336_basics.loss_optimizer.loss_opt import cross_entropy_loss, AdamW, lr_cosine_schedule, gradient_clipping


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on a dataset")

    # 分组命令
    io_args = parser.add_argument_group('数据路径参数')
    io_args.add_argument('--train_data_path', type=str,
                         default='train.bin', help='训练集路径')
    io_args.add_argument('--valid_data_path', type=str,
                         default='valid.bin', help='验证集路径')
    io_args.add_argument('--checkpoint_dir', type=str,
                         default='checkpoints', help='检查点目录')

    model_args = parser.add_argument_group('模型参数')
    model_args.add_argument('--vocab_size', type=int, default=50257,
                            help='词汇表大小')
    model_args.add_argument('--context_length', type=int, default=256,
                            help='上下文长度')
    model_args.add_argument('--d_model', type=int, default=256,
                            help='模型维度')
    model_args.add_argument('--num_heads', type=int, default=8,
                            help='头数')
    model_args.add_argument('--num_layers', type=int, default=6,
                            help='Transformer block 数')
    model_args.add_argument('--d_ff', type=int, default=1024,
                            help='前馈网络中间层维度')

    model_args.add_argument('--rope_theta', type=float, default=10000.0,
                            help='RoPE 旋转角度')

    training_args = parser.add_argument_group('训练参数')
    training_args.add_argument('--batch_size', type=int, default=16,
                               help='批次大小')
    training_args.add_argument('--max_iters', type=int, default=1000,
                               help='最大训练迭代次数')
    training_args.add_argument('--lr', type=float, default=6e-4,
                               help='学习率')
    training_args.add_argument('--weight_decay', type=float, default=1e-1,
                               help='权重衰减')
    training_args.add_argument('--warmup_iters', type=int, default=100,
                               help='预热迭代次数')
    training_args.add_argument('--grad_clip', type=float, default=1.0,
                               help='梯度裁剪值')

    runtime_args = parser.add_argument_group('运行与日志参数')
    runtime_args.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                              help='运行设备')
    runtime_args.add_argument('--log_interval', type=int, default=10,
                              help='日志打印间隔')
    runtime_args.add_argument('--eval_interval', type=int, default=100,
                              help='验证间隔')
    runtime_args.add_argument('--eval_iters', type=int, default=100,
                              help='验证迭代次数')
    return parser.parse_args()


def generate_dummy_data_if_needed(train_path, val_path, vocab_size, context_length):
    """如果数据文件不存在，则创建虚拟的token ID序列文件。"""
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("--- 未找到数据文件，正在创建虚拟的 train.bin 和 val.bin ---")
        train_size = max(50000, context_length * 10)
        val_size = max(5000, context_length * 2)

        train_ids = np.random.randint(
            0, vocab_size, size=train_size, dtype=np.uint16)
        train_ids.tofile(train_path)

        val_ids = np.random.randint(
            0, vocab_size, size=val_size, dtype=np.uint16)
        val_ids.tofile(val_path)
        print(f"✅ 已成功创建 {train_path} 和 {val_path}")


def log_model_parameters(weights: dict[str, torch.Tensor]):
    """计算并打印模型的总参数量。"""
    total_params = sum(p.numel() for p in weights.values())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")


def calculate_grad_norm(parameters) -> float:
    """在梯度裁剪前计算所有参数的总梯度范数。"""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


@torch.no_grad()
def estimate_loss(model, config, train_data, val_data):
    out = {}
    model.eval()
    for split, data in {'train': train_data, 'val': val_data}.items():
        losses = torch.zeros(config.eval_iters, device=config.device)
        for k in range(config.eval_iters):
            X, Y = get_batch(data, config.batch_size,
                             config.context_length, config.device)
            logits = model(X)
            loss = cross_entropy_loss(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # 恢复为训练模式
    return out


def initialize_weights(config: argparse.Namespace) -> dict[str, torch.Tensor]:
    """初始化模型权重。"""
    weights = {}

    def create_tensor(*shape):
        return torch.randn(shape, device=config.device)

    weights['embeddings.weight'] = create_tensor(
        config.vocab_size, config.d_model) * 0.02
    weights['norm.weight'] = torch.ones(config.d_model, device=config.device)
    weights['proj.weight'] = create_tensor(config.vocab_size, config.d_model)

    for i in range(config.num_layers):
        prefix = f'layers.{i}.'
        for name in ['attn.q_proj.weight', 'attn.k_proj.weight', 'attn.v_proj.weight', 'attn.out_proj.weight']:
            weights[prefix + name] = create_tensor(
                config.d_model, config.d_model) * 0.02
        for name in ['norm1.weight', 'norm2.weight']:
            weights[prefix + name] = torch.ones(
                config.d_model, device=config.device)
        # 注意：Linear层权重的维度顺序应为(out_features, in_features)
        weights[prefix + 'ffn.linear1.weight'] = create_tensor(
            config.d_ff, config.d_model)
        weights[prefix + 'ffn.linear2.weight'] = create_tensor(
            config.d_model, config.d_ff)
        weights[prefix + 'ffn.linear3.weight'] = create_tensor(
            config.d_ff, config.d_model)
    for w in weights.values():
        w.requires_grad_(True)
    return weights


def load_weights(model, weights, config):
    model.embedding.weight.data.copy_(weights['embeddings.weight'])

    for layer_idx in range(config.num_layers):
        block = model.blocks[layer_idx]
        prefix = f"layers.{layer_idx}."
        block.attn.q_proj.weight.data.copy_(
            weights[prefix + "attn.q_proj.weight"])
        block.attn.k_proj.weight.data.copy_(
            weights[prefix + "attn.k_proj.weight"])
        block.attn.v_proj.weight.data.copy_(
            weights[prefix + "attn.v_proj.weight"])
        block.attn.out_proj.weight.data.copy_(
            weights[prefix + "attn.out_proj.weight"])
        block.norm1.weight.data.copy_(weights[prefix + "norm1.weight"])
        block.ffn.linear1.weight.data.copy_(
            weights[prefix + "ffn.linear1.weight"])
        block.ffn.linear2.weight.data.copy_(
            weights[prefix + "ffn.linear2.weight"])
        block.ffn.linear3.weight.data.copy_(
            weights[prefix + "ffn.linear3.weight"])
        block.norm2.weight.data.copy_(weights[prefix + "norm2.weight"])

    model.norm.weight.data.copy_(weights["norm.weight"])
    model.proj.weight.data.copy_(weights["proj.weight"])


def main():
    config = get_config()
    generate_dummy_data_if_needed(
        config.train_data_path, config.valid_data_path, config.vocab_size, config.context_length)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.manual_seed(42)
    if 'cuda' in config.device:
        torch.cuda.manual_seed(42)
    ctx = torch.amp.autocast(device_type=config.device.split(
        ':')[0], dtype=torch.bfloat16) if 'cuda' in config.device else nullcontext()

    print(f"正在从 {config.train_data_path} 和 {config.valid_data_path} 加载数据...")
    train_data = np.memmap(config.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(config.valid_data_path, dtype=np.uint16, mode='r')

    print("正在初始化模型权重...")
    weights = initialize_weights(config)

    print("\n" + "="*50)
    print("训练配置:")
    for key, value in sorted(vars(config).items()):
        print(f"  {key:<20}: {value}")
    print("="*50 + "\n")

    log_model_parameters(weights)
    print("\n")

    optimizer = AdamW(weights.values(), lr=config.lr, betas=(
        0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay)

    print("开始训练...")
    pbar = tqdm(range(config.max_iters), desc="训练进度")
    model = TransfromerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
        dtype=next(iter(weights.values())).dtype,
    )

    load_weights(model, weights, config)
    for iter_num in pbar:
        # 更新学习率
        lr = lr_cosine_schedule(
            iter_num, config.lr, 0.1*config.lr, config.warmup_iters, config.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向和后向传播
        X, Y = get_batch(train_data, config.batch_size,
                         config.context_length, config.device)
        with ctx:
            logits = model(X)
            loss = cross_entropy_loss(logits, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = calculate_grad_norm(model.parameters())
        gradient_clipping(model.parameters(), config.grad_clip)

        optimizer.step()

        # 定期评估与保存检查点
        if iter_num > 0 and iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config, train_data, val_data)

            tqdm.write("\n" + "-"*50)
            tqdm.write(f"步数 {iter_num}:")
            tqdm.write(f"  - 学习率: {lr:.6f}")
            tqdm.write(
                f"  - 训练集损失: {losses['train']:.4f}, 验证集损失: {losses['val']:.4f}")
            tqdm.write(f"  - 梯度范数 (裁剪前): {grad_norm:.4f}")
            tqdm.write(
                f"  - Logits 统计: Mean={logits.mean():.2f}, Std={logits.std():.2f}, Min={logits.min():.2f}, Max={logits.max():.2f}")
            tqdm.write(f"  - 输入样本 (前10个token): {X[0, :10].tolist()}")
            tqdm.write("-"*50)

            checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state': optimizer.state_dict(
            ), 'iter_num': iter_num, 'config': config}
            chkpt_path = os.path.join(
                config.checkpoint_dir, f'ckpt_{iter_num}.pt')
            tqdm.write(f"正在保存检查点到 {chkpt_path}\n")
            torch.save(checkpoint, chkpt_path)

        # 更新进度条信息
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{lr:.6f}',
            'grad_norm': f'{grad_norm:.2f}'
        })


if __name__ == '__main__':
    main()
