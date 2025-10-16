#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>

struct CompareHeapElement {
    bool operator()(const std::tuple<int, std::string, std::string>& a, 
                   const std::tuple<int, std::string, std::string>& b) const {
        int countA = std::get<0>(a);
        int countB = std::get<0>(b);
        
        // 首先按count降序
        if (countA != countB) {
            return countA < countB;
        }
        
        // 当count相同时，按照两个字符串拼接后的结果进行降序排序
        const std::string& firstA = std::get<1>(a);
        const std::string& secondA = std::get<2>(a);
        const std::string& firstB = std::get<1>(b);
        const std::string& secondB = std::get<2>(b);
        
        std::string combinedA = firstA + secondA;
        std::string combinedB = firstB + secondB;
        
        return combinedA < combinedB;
    }
};


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
  std::priority_queue<std::tuple<int, std::string, std::string>,
                      std::vector<std::tuple<int, std::string, std::string>>,
                      CompareHeapElement> heap;
  // 记录堆中每个字节对的信息，快速访问
  std::unordered_map<std::pair<std::string, std::string>,
                     std::shared_ptr<std::tuple<int, std::string, std::string>>,
                     PairHash>
      heap_entries;

public:
  BPEIndex(const std::vector<std::vector<std::string>> &sequences);
  // 修改前
  //   std::pair<std::string, std::string> get_most_frequent_pair();

  // 修改后
  std::optional<std::pair<std::string, std::string>> get_most_frequent_pair();
  int merge_pair(const std::pair<std::string, std::string> &pair,
                 const std::string new_token);

private:
  void add_position(const std::pair<std::string, std::string> &pair,
                    int seq_idx, int pos);
  void update_pair_count(const std::pair<std::string, std::string> &pair,
                         int delta);
};