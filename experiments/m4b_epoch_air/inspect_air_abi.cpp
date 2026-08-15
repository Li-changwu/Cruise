#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "graph/graph.h"

namespace {
std::string ToString(const ge::AscendString &value) {
  const char *text = value.GetString();
  return text == nullptr ? std::string() : std::string(text);
}

void PrintShape(const ge::TensorDesc &desc) {
  const std::vector<int64_t> dims = desc.GetShape().GetDims();
  std::cout << '[';
  for (size_t index = 0; index < dims.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << dims[index];
  }
  std::cout << ']';
}

struct DataNode {
  int64_t index;
  std::string name;
  ge::DataType dtype;
  ge::TensorDesc desc;
};
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " AIR\n";
    return 2;
  }
  ge::Graph graph("CruiseM4bAirAbiInspector");
  if (graph.LoadFromFile(argv[1]) != ge::GRAPH_SUCCESS || !graph.IsValid()) {
    return 3;
  }

  std::vector<DataNode> data_nodes;
  const ge::AscendString index_name("index");
  for (const ge::GNode &node : graph.GetAllNodes()) {
    ge::AscendString type;
    if (node.GetType(type) != ge::GRAPH_SUCCESS || ToString(type) != "Data") {
      continue;
    }
    ge::AscendString name;
    ge::TensorDesc desc;
    int64_t index = -1;
    if (node.GetName(name) != ge::GRAPH_SUCCESS ||
        node.GetOutputDesc(0, desc) != ge::GRAPH_SUCCESS ||
        node.GetAttr(index_name, index) != ge::GRAPH_SUCCESS) {
      return 4;
    }
    data_nodes.push_back({index, ToString(name), desc.GetDataType(), desc});
  }
  std::sort(data_nodes.begin(), data_nodes.end(),
            [](const DataNode &left, const DataNode &right) {
              return left.index < right.index;
            });
  for (const DataNode &node : data_nodes) {
    std::cout << node.index << '\t' << node.name << '\t'
              << static_cast<int>(node.dtype) << '\t';
    PrintShape(node.desc);
    std::cout << '\n';
  }
  return data_nodes.empty() ? 5 : 0;
}
