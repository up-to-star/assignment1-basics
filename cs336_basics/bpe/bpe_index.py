from typing import List, DefaultDict, Tuple, Dict, Any
from collections import defaultdict
import heapq


class BPEIndex:

    def __init__(self, sequences: List[List[str]]):
        self.sequences = sequences
        self.pair_counts: DefaultDict[Tuple[str,
                                            str], int] = defaultdict(int)  # 统计字节对数
        self.pair_positions: DefaultDict[Tuple[str, str],
                                         List[Tuple[int, int]]] = defaultdict(list)  # 记录字节对出现的位置
        self.heap = []  # 最大堆，存储最高频字节对
        self.heap_entries: Dict[Tuple[str, str], Any] = {}  # 记录堆中每个字节对的信息，快速访问

        for seq_idx, seq in enumerate(sequences):
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                self.pair_counts[pair] += 1
                self.pair_positions[pair].append((seq_idx, i))

        for pair, count in self.pair_counts.items():
            entry = [-count, pair]
            heapq.heappush(self.heap, entry)
            self.heap_entries[pair] = entry

    def get_most_frequent_pair(self) -> Tuple[str, str]:
        while self.heap:
            neg_count, pair = self.heap[0]
            if pair not in self.heap_entries:
                heapq.heappop(self.heap)
                continue

            current_count = self.pair_counts.get(pair, 0)
            if current_count == -neg_count and current_count > 1:
                return pair
            heapq.heappop(self.heap)
            if pair in self.heap_entries:
                del self.heap_entries[pair]
        return None
