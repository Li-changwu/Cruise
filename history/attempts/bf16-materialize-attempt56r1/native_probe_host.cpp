#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ge/ge_api.h"
#include "graph/graph.h"

namespace {
constexpr size_t kElements = 18944U;
constexpr size_t kBytes = kElements * sizeof(uint16_t);

bool ReadFile(const std::string &path, std::vector<uint8_t> &data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || static_cast<size_t>(stream.tellg()) != kBytes) return false;
  data.resize(kBytes);
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(data.data()), kBytes);
  return static_cast<bool>(stream);
}

ge::Tensor MakeTensor(std::vector<uint8_t> &data) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape({1, 1, 18944}), ge::FORMAT_ND, ge::DT_BF16));
  tensor.SetData(data.data(), data.size());
  return tensor;
}

bool WriteFile(const std::string &path, const ge::Tensor &tensor) {
  if (tensor.GetData() == nullptr || tensor.GetSize() != kBytes) return false;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return false;
  stream.write(reinterpret_cast<const char *>(tensor.GetData()), kBytes);
  return static_cast<bool>(stream);
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: native_probe_host AIR INPUT OUTPUT\n";
    return 2;
  }
  ge::Graph graph("Bf16MaterializeProbe");
  if (graph.LoadFromFile(argv[1]) != ge::GRAPH_SUCCESS || !graph.IsValid()) return 3;
  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) return static_cast<int>(ret);
  auto session = std::make_shared<ge::Session>(std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, graph);
  if (ret != ge::SUCCESS) { ge::GEFinalize(); return 4; }
  std::vector<uint8_t> input;
  if (!ReadFile(argv[2], input)) { ge::GEFinalize(); return 5; }
  std::vector<ge::Tensor> inputs = {MakeTensor(input)};
  std::vector<ge::Tensor> outputs;
  ret = session->RunGraph(0, inputs, outputs);
  if (ret != ge::SUCCESS || outputs.size() != 1 || !WriteFile(argv[3], outputs[0])) {
    ge::GEFinalize();
    return 6;
  }
  ge::GEFinalize();
  std::cout << "BF16_MATERIALIZE_NATIVE_COMPLETE" << std::endl;
  return 0;
}
