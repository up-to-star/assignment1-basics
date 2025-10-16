#include "bpeindex.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include <optional>

namespace py = pybind11;

PYBIND11_MODULE(bpeindex_cpp, m) {
  m.doc() = "BPE Index C++ Extension";
  py::class_<BPEIndex>(m, "BPEIndex")
      .def(py::init<const std::vector<std::vector<std::string>> &>())
      .def("get_most_frequent_pair", &BPEIndex::get_most_frequent_pair)
      .def("merge_pair", &BPEIndex::merge_pair);
}