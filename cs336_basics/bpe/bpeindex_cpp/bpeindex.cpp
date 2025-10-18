#include "bpeindex.h"
#include <memory>

BPEIndex::BPEIndex(const std::vector<std::vector<std::string>> &sequences)
    : sequences(sequences) {
  for (int seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
    const auto &seq = sequences[seq_idx];
    for (int pos = 0; pos < seq.size() - 1; ++pos) {
      std::pair<std::string, std::string> pair = {seq[pos], seq[pos + 1]};
      pair_counts[pair]++;
      pair_positions[pair].emplace_back(seq_idx, pos);
    }
  }

  // 初始化最大堆
  for (const auto &[pair, count] : pair_counts) {
    if (count > 1) {
      auto entry = std::make_shared<std::tuple<int, std::string, std::string>>(
          count, pair.first, pair.second);
      heap.push(*entry);
      heap_entries[pair] = entry;
    }
  }
}

void BPEIndex::add_position(const std::pair<std::string, std::string> &pair,
                            int seq_idx, int pos) {
  pair_positions[pair].emplace_back(seq_idx, pos);
}

void BPEIndex::update_pair_count(
    const std::pair<std::string, std::string> &pair, int delta) {
  if (delta == 0) {
    return;
  }

  if (!pair_counts.count(pair)) {
    pair_counts[pair] = 0;
  }
  int new_count = pair_counts[pair] + delta;
  pair_counts[pair] = new_count;
  if (new_count < 0) {
    new_count = 0;
    pair_counts[pair] = new_count;
  }

  // 更新堆条目，但不修改现有堆
  if (heap_entries.count(pair)) {
    // 仅更新heap_entries，不修改堆
    // 旧的条目会在get_most_frequent_pair中被过滤掉
    auto entry = std::make_shared<std::tuple<int, std::string, std::string>>(new_count, pair.first, pair.second);
    heap_entries[pair] = entry;
  } else if (new_count > 1) {
    // 只有当新计数大于1时，才将新条目添加到堆中
    auto entry = std::make_shared<std::tuple<int, std::string, std::string>>(new_count, pair.first, pair.second);
    heap.push(*entry);
    heap_entries[pair] = entry;
  }
}

std::optional<std::pair<std::string, std::string>> BPEIndex::get_most_frequent_pair() {
  while (!heap.empty()) {
    auto [count, first, second] = heap.top();
    std::pair<std::string, std::string> pair = {first, second};
    
    // 检查该对是否在heap_entries中，且当前计数是否与堆中计数一致
    if (heap_entries.count(pair)) {
      int current_count = pair_counts.count(pair) ? pair_counts[pair] : 0;
      auto& current_entry = heap_entries[pair];
      int entry_count = std::get<0>(*current_entry);
      
      // 如果当前计数大于1且与heap_entries中的计数一致，则返回该对
      if (current_count > 1 && current_count == entry_count) {
        return pair;
      }
    }
    
    // 否则弹出堆顶元素
    heap.pop();
  }
  
  // 检查是否还有有效对（即使不在堆中）
  for (const auto& [pair, count] : pair_counts) {
    if (count > 1) {
      // 如果有有效对但不在堆中，将其添加到堆中
      auto entry = std::make_shared<std::tuple<int, std::string, std::string>>(count, pair.first, pair.second);
      heap.push(*entry);
      heap_entries[pair] = entry;
      return pair;
    }
  }
  
  return std::nullopt;
}

int BPEIndex::merge_pair(const std::pair<std::string, std::string> &pair,
                         const std::string new_token) {
  if (!pair_counts.count(pair) || pair_counts[pair] == 0) {
    return 0;
  }
  std::unordered_map<int, std::vector<int>> positions_by_seq;
  for (const auto &[seq_idx, pos] : pair_positions[pair]) {
    positions_by_seq[seq_idx].push_back(pos);
  }

  int merge_count = 0;
  for (const auto &[seq_idx, positions] : positions_by_seq) {
    auto &seq = sequences[seq_idx];
    std::vector<int> sorted_positions = positions;
    std::sort(sorted_positions.rbegin(), sorted_positions.rend());
    int last_merged_pos = -2;
    for (int pos : sorted_positions) {
      if (pos >= seq.size() - 1 || pos <= last_merged_pos) {
        continue;
      }

      if (seq[pos] != pair.first || seq[pos + 1] != pair.second) {
        continue;
      }
      seq[pos] = new_token;
      seq.erase(seq.begin() + pos + 1);
      last_merged_pos = pos;
      merge_count++;

      if (pos > 0) {
        std::pair<std::string, std::string> left_pair = {seq[pos - 1],
                                                         pair.first};
        update_pair_count(left_pair, -1);
        std::pair<std::string, std::string> new_left_pair = {seq[pos - 1],
                                                             new_token};
        update_pair_count(new_left_pair, 1);
        add_position(new_left_pair, seq_idx, pos - 1);
      }
      if (pos < seq.size() - 1) {
        std::pair<std::string, std::string> right_pair = {pair.second,
                                                          seq[pos + 1]};
        update_pair_count(right_pair, -1);
        std::pair<std::string, std::string> new_right_pair = {new_token,
                                                              seq[pos + 1]};
        update_pair_count(new_right_pair, 1);
        add_position(new_right_pair, seq_idx, pos);
      }
    }
  }
  if (pair_counts.count(pair)) {
    pair_counts.erase(pair);
  }

  if (pair_positions.count(pair)) {
    pair_positions.erase(pair);
  }
  if (heap_entries.count(pair)) {
    heap_entries.erase(pair);
  }
  return merge_count;
}