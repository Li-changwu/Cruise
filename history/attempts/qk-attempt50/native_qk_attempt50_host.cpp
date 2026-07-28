#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ge/ge_api.h"
#include "graph/graph.h"

namespace {
constexpr size_t kABytes = 1U * 28U * 128U * sizeof(uint16_t);
constexpr size_t kBBytes = 28U * 128U * 8U * sizeof(uint16_t);
constexpr size_t kTilingBytes = 18U * sizeof(uint32_t);
constexpr size_t kRawBytes = 1U * 28U * 8U * sizeof(uint16_t);
constexpr size_t kScaledBytes = 1U * 28U * 1U * 8U * sizeof(float);

bool ReadFile(const std::string &path, size_t expected, std::vector<uint8_t> &data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const auto size = static_cast<size_t>(stream.tellg());
  if (size != expected) return false;
  data.resize(size);
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
  return static_cast<bool>(stream);
}

ge::Tensor MakeTensor(std::vector<uint8_t> &data, const std::vector<int64_t> &shape,
                      ge::DataType dtype) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype));
  tensor.SetData(data.data(), data.size());
  return tensor;
}

std::string StepPath(const std::string &directory, int step, const std::string &name) {
  std::ostringstream stream;
  stream << directory << "/step" << step << "_" << name << ".bin";
  return stream.str();
}

bool WriteOutput(const std::string &path, const ge::Tensor &tensor, size_t expected) {
  if (tensor.GetSize() != expected || tensor.GetData() == nullptr) return false;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return false;
  stream.write(reinterpret_cast<const char *>(tensor.GetData()),
               static_cast<std::streamsize>(tensor.GetSize()));
  return static_cast<bool>(stream);
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: native_qk_attempt50_host AIR INPUT_DIR OUTPUT_DIR CACHE_DIR\n";
    return 2;
  }
  ge::Graph graph("G4aQkAttempt50");
  const auto load_status = graph.LoadFromFile(argv[1]);
  std::cout << "G4A_QK_ATTEMPT50_LOAD status=" << load_status
            << " valid=" << graph.IsValid() << std::endl;
  if (load_status != ge::GRAPH_SUCCESS || !graph.IsValid()) return 3;

  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
      {"ge.op_compiler_cache_mode", "enable"},
      {"ge.op_compiler_cache_dir", argv[4]}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) return static_cast<int>(ret);
  auto session = std::make_shared<ge::Session>(std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, graph);
  if (ret != ge::SUCCESS) {
    ge::GEFinalize();
    return 4;
  }

  std::vector<uint8_t> tiling;
  if (!ReadFile(std::string(argv[2]) + "/tiling.bin", kTilingBytes, tiling)) {
    ge::GEFinalize();
    return 5;
  }
  const char *names[] = {"raw_qk", "legacy_scaled", "fp32_div_bf16", "fp32_mul_bf16"};
  const size_t sizes[] = {kRawBytes, kScaledBytes, kScaledBytes, kScaledBytes};

  for (int step = 1; step <= 4; ++step) {
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    if (!ReadFile(StepPath(argv[2], step, "a"), kABytes, a) ||
        !ReadFile(StepPath(argv[2], step, "b"), kBBytes, b)) {
      ge::GEFinalize();
      return 5;
    }
    std::vector<ge::Tensor> inputs = {
        MakeTensor(a, {1, 28, 128}, ge::DT_BF16),
        MakeTensor(b, {28, 128, 8}, ge::DT_BF16),
        MakeTensor(tiling, {72}, ge::DT_UINT8)};
    std::vector<ge::Tensor> outputs;
    ret = session->RunGraph(0, inputs, outputs);
    if (ret != ge::SUCCESS || outputs.size() != 4) {
      std::cerr << "RunGraph failed step=" << step << " ret=" << ret
                << " outputs=" << outputs.size() << std::endl;
      ge::GEFinalize();
      return 6;
    }
    for (size_t i = 0; i < 4; ++i) {
      if (!WriteOutput(StepPath(argv[3], step, names[i]), outputs[i], sizes[i])) {
        ge::GEFinalize();
        return 7;
      }
    }
    std::cout << "G4A_QK_ATTEMPT50_STEP step=" << step << " ret=" << ret << std::endl;
  }
  ge::GEFinalize();
  std::cout << "G4A_QK_ATTEMPT50_NATIVE_COMPLETE" << std::endl;
  return 0;
}

