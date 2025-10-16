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

    def merge_pair(self, pair: Tuple[str, str], new_token: str) -> int:
        if pair not in self.pair_positions or not self.pair_positions[pair]:
            return 0

        positions_by_seq = defaultdict(list)
        for seq_idx, pos in self.pair_positions[pair]:
            positions_by_seq[seq_idx].append(pos)

        merge_count = 0
        for seq_idx, positions in positions_by_seq.items():
            seq = self.sequences[seq_idx]
            positions.sort(reverse=True)
            last_merged_pos = -2

            for pos in positions:
                if pos >= len(seq) - 1 or pos <= last_merged_pos:
                    continue

                if seq[pos] != pair[0] or seq[pos + 1] != pair[1]:
                    continue

                seq[pos] = new_token
                del seq[pos + 1]
                merge_count += 1
                last_merged_pos = pos

                if pos > 0:
                    left_pair = (seq[pos - 1], pair[0])
                    self._update_pair_count(left_pair, -1)

                    new_left_pair = (seq[pos - 1], new_token)
                    self._update_pair_count(new_left_pair, 1)
                    self._add_position(new_left_pair, seq_idx, pos - 1)

                if pos < len(seq) - 1:
                    right_pair = (pair[1], seq[pos + 1])
                    self._update_pair_count(right_pair, -1)

                    new_right_pair = (new_token, seq[pos + 1])
                    self._update_pair_count(new_right_pair, 1)
                    self._add_position(new_right_pair, seq_idx, pos)
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

        if pair in self.heap_entries and self.heap_entries[pair] is not None:
            self.heap_entries[pair][0] = -new_count
            heapq.heapify(self.heap)
        elif new_count > 1:
            new_entry = [-new_count, pair]
            heapq.heappush(self.heap, new_entry)
            self.heap_entries[pair] = new_entry
