#pragma once

#include <algorithm>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct PairHash {
  template <typename T1, typename T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>()(p.first);
    auto h2 = std::hash<T2>()(p.second);
    return h1 ^ (h2 << 1);
  }
};

class BPEIndex {
private:
  std::vector<std::vector<std::string>> sequences;
  // 统计字节对数
  std::unordered_map<std::pair<std::string, std::string>, int, PairHash>
      pair_counts;
  // 记录字节对出现的位置
  std::unordered_map<std::pair<std::string, std::string>,
                     std::vector<std::pair<int, int>>, PairHash>
      pair_positions;
  // 最大堆，用于存储字节对出现次数和字节对
  std::priority_queue<std::tuple<int, std::string, std::string>> heap;
  // 记录堆中每个字节对的信息，快速访问
  std::unordered_map<std::pair<std::string, std::string>,
                     std::shared_ptr<std::tuple<int, std::string, std::string>>,
                     PairHash>
      heap_entries;

public:
  BPEIndex(const std::vector<std::vector<std::string>> &sequences);
  std::pair<std::string, std::string> get_most_frequent_pair();
  int merge_pair(const std::pair<std::string, std::string> &pair,
                 const std::string new_token);

private:
  void add_position(const std::pair<std::string, std::string> &pair,
                    int seq_idx, int pos);
  void update_pair_count(const std::pair<std::string, std::string> &pair,
                         int delta);
};