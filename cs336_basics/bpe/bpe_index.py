from typing import List, DefaultDict, Tuple, Dict, Any
from collections import defaultdict
import heapq


class BPEIndex:

    def __init__(self, sequences: List[List[str]]):
        self.sequences = sequences
        self.pair_counts: DefaultDict[Tuple[str,
                                            # 统计字节对数
                                            str], int] = defaultdict(int)
        self.pair_positions: DefaultDict[Tuple[str, str],
                                         # 记录字节对出现的位置
                                         List[Tuple[int, int]]] = defaultdict(list)
        self.heap = []  # 最大堆，存储最高频字节对
        self.heap_entries: Dict[Tuple[str, str], Any] = {}  # 记录堆中每个字节对的信息，快速访问

        for seq_idx, seq in enumerate(sequences):
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                self.pair_counts[pair] += 1
                self.pair_positions[pair].append((seq_idx, i))

        for pair, count in self.pair_counts.items():
            if count > 1:
                entry = [-count, pair]
                heapq.heappush(self.heap, entry)
                self.heap_entries[pair] = entry

    def get_most_frequent_pair(self) -> Tuple[str, str]:
        while self.heap:
            # 获取堆顶元素，但不移除
            neg_count, pair = self.heap[0]
            
            # 检查该元素是否仍然有效
            if pair not in self.heap_entries or self.heap_entries[pair] is None:
                heapq.heappop(self.heap)  # 移除无效元素
                continue

            # 获取当前实际计数
            current_count = self.pair_counts.get(pair, 0)
            
            # 检查计数是否匹配且有效（大于1）
            if current_count > 1:
                # 如果计数已更新，需要重新插入堆
                if current_count != -neg_count:
                    heapq.heappop(self.heap)  # 移除旧条目
                    self.heap_entries[pair] = None
                    # 添加更新后的条目
                    new_entry = [-current_count, pair]
                    heapq.heappush(self.heap, new_entry)
                    self.heap_entries[pair] = new_entry
                    continue  # 继续循环，下一次迭代会返回更新后的堆顶
                return pair
            
            # 计数小于等于1，移除该条目
            heapq.heappop(self.heap)
            if pair in self.heap_entries:
                self.heap_entries[pair] = None
        return None

    def merge_pair(self, pair: Tuple[str, str], new_token: str) -> int:
        if pair not in self.pair_positions or not self.pair_positions[pair]:
            return 0

        positions_by_seq = defaultdict(list)
        for seq_idx, pos in self.pair_positions[pair]:
            positions_by_seq[seq_idx].append(pos)

        merge_count = 0
        for seq_idx, positions in positions_by_seq.items():
            seq = self.sequences[seq_idx]
            # 按逆序处理，避免位置偏移
            positions.sort(reverse=True)
            last_merged_pos = -2

            for pos in positions:
                # 检查位置有效性和合并状态
                if pos >= len(seq) - 1 or pos <= last_merged_pos:
                    continue

                # 再次验证字节对是否匹配（防止并发修改）
                if seq[pos] != pair[0] or seq[pos + 1] != pair[1]:
                    continue

                # 执行合并
                seq[pos] = new_token
                del seq[pos + 1]
                merge_count += 1
                last_merged_pos = pos

                # 更新左侧相邻字节对
                if pos > 0:
                    left_pair_old = (seq[pos - 1], pair[0])
                    self._update_pair_count(left_pair_old, -1)
                    
                    left_pair_new = (seq[pos - 1], new_token)
                    self._update_pair_count(left_pair_new, 1)
                    self._add_position(left_pair_new, seq_idx, pos - 1)

                # 更新右侧相邻字节对
                if pos < len(seq) - 1:
                    right_pair_old = (pair[1], seq[pos + 1])
                    self._update_pair_count(right_pair_old, -1)
                    
                    right_pair_new = (new_token, seq[pos + 1])
                    self._update_pair_count(right_pair_new, 1)
                    self._add_position(right_pair_new, seq_idx, pos)

        # 清理旧的字节对数据
        if pair in self.pair_counts:
            del self.pair_counts[pair]
        if pair in self.pair_positions:
            del self.pair_positions[pair]
        if pair in self.heap_entries:
            self.heap_entries[pair] = None
        return merge_count

    def _add_position(self, pair: Tuple[str, str], seq_idx: int, pos: int):
        self.pair_positions[pair].append((seq_idx, pos))

    def _update_pair_count(self, pair: Tuple[str, str], delta: int):
        if delta == 0:
            return
        if pair not in self.pair_counts:
            self.pair_counts[pair] = 0

        new_count = self.pair_counts[pair] + delta
        self.pair_counts[pair] = new_count

        if new_count < 0:
            new_count = 0
            self.pair_counts[pair] = 0

        # 修复堆更新逻辑，避免使用heapify
        if pair in self.heap_entries and self.heap_entries[pair] is not None:
            # 移除旧条目标记
            self.heap_entries[pair] = None
            # 只有当新计数大于1时添加新条目
            if new_count > 1:
                new_entry = [-new_count, pair]
                heapq.heappush(self.heap, new_entry)
                self.heap_entries[pair] = new_entry
        elif new_count > 1:
            new_entry = [-new_count, pair]
            heapq.heappush(self.heap, new_entry)
            self.heap_entries[pair] = new_entry
