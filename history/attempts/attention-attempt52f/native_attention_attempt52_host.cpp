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
constexpr size_t kHiddenBytes = 4U * 3584U * sizeof(uint16_t);
constexpr size_t kCacheBytes = 1U * 4U * 8U * 128U * sizeof(uint16_t);
constexpr size_t kPositionBytes = sizeof(int64_t);
constexpr size_t kTilingBytes = 72U;
constexpr size_t kAttentionBytes = 1U * 1U * 3584U * sizeof(uint16_t);
constexpr size_t kKProjectionBytes = 1U * 1U * 512U * sizeof(uint16_t);
constexpr size_t kKRoPEBytes = 1U * 4U * 1U * 128U * sizeof(uint16_t);
constexpr size_t kQKScoresBytes = 1U * 28U * 1U * 8U * sizeof(float);
constexpr size_t kQKScoresBf16Bytes = 1U * 28U * 1U * 8U * sizeof(uint16_t);
constexpr size_t kQProjectionBytes = 1U * 1U * 3584U * sizeof(uint16_t);
constexpr size_t kQRoPEBytes = 1U * 28U * 1U * 128U * sizeof(uint16_t);
constexpr size_t kRoPECoefficientBytes = 1U * 1U * 1U * 128U * sizeof(uint16_t);

bool ReadFile(const std::string &path, size_t expected, std::vector<uint8_t> &data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const auto size = static_cast<size_t>(stream.tellg());
  if (size != expected) {
    std::cerr << "size mismatch path=" << path << " actual=" << size
              << " expected=" << expected << std::endl;
    return false;
  }
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

bool WriteTensor(const std::string &path, const ge::Tensor &tensor, size_t expected) {
  if (tensor.GetData() == nullptr || tensor.GetSize() != expected) {
    std::cerr << "invalid output path=" << path << " size=" << tensor.GetSize()
              << " expected=" << expected << std::endl;
    return false;
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return false;
  stream.write(reinterpret_cast<const char *>(tensor.GetData()),
               static_cast<std::streamsize>(tensor.GetSize()));
  return static_cast<bool>(stream);
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: native_attention_attempt52_host AIR INPUT_DIR TILING OUTPUT_DIR CACHE_DIR\n";
    return 2;
  }
  ge::Graph graph("G4aAttentionAttempt52");
  const auto load_status = graph.LoadFromFile(argv[1]);
  std::cout << "G4A_ATTEMPT52_LOAD status=" << load_status
            << " valid=" << graph.IsValid() << std::endl;
  if (load_status != ge::GRAPH_SUCCESS || !graph.IsValid()) return 3;
  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
      {"ge.op_compiler_cache_mode", "enable"},
      {"ge.op_compiler_cache_dir", argv[5]}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) return static_cast<int>(ret);
  auto session = std::make_shared<ge::Session>(std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, graph);
  if (ret != ge::SUCCESS) {
    ge::GEFinalize();
    return 4;
  }

  std::vector<uint8_t> hidden, tiling;
  if (!ReadFile(std::string(argv[2]) + "/hidden.bin", kHiddenBytes, hidden) ||
      !ReadFile(argv[3], kTilingBytes, tiling)) {
    ge::GEFinalize();
    return 5;
  }
  const std::vector<std::string> names = {
      "attention", "key_cache", "value_cache", "position", "k_projection",
      "k_rope", "qk_scores", "q_projection", "q_rope", "rope_cos", "rope_sin",
      "q_projection_bf16", "k_projection_bf16", "q_rope_bf16", "k_rope_bf16",
      "updated_key_bf16", "qk_scores_bf16", "masked_scores_bf16",
      "probabilities_bf16", "attention_value_bf16", "attention_flat_bf16",
      "attention_output_bf16"};
  const std::vector<size_t> sizes = {
      kAttentionBytes, kCacheBytes, kCacheBytes, kPositionBytes,
      kKProjectionBytes, kKRoPEBytes, kQKScoresBytes, kQProjectionBytes,
      kQRoPEBytes, kRoPECoefficientBytes, kRoPECoefficientBytes,
      kQProjectionBytes, kKProjectionBytes, kQRoPEBytes, kKRoPEBytes,
      kCacheBytes, kQKScoresBf16Bytes, kQKScoresBf16Bytes, kQKScoresBf16Bytes,
      kQRoPEBytes, kAttentionBytes, kAttentionBytes};

  for (int step = 1; step <= 4; ++step) {
    std::vector<uint8_t> key, value, position;
    if (!ReadFile(StepPath(argv[2], step, "key_cache"), kCacheBytes, key) ||
        !ReadFile(StepPath(argv[2], step, "value_cache"), kCacheBytes, value) ||
        !ReadFile(StepPath(argv[2], step, "position"), kPositionBytes, position)) {
      ge::GEFinalize();
      return 6;
    }
    std::vector<ge::Tensor> inputs = {
        MakeTensor(key, {1, 4, 8, 128}, ge::DT_FLOAT16),
        MakeTensor(hidden, {4, 3584}, ge::DT_FLOAT16),
        MakeTensor(position, {1}, ge::DT_INT64),
        MakeTensor(value, {1, 4, 8, 128}, ge::DT_FLOAT16),
        MakeTensor(tiling, {72}, ge::DT_UINT8)};
    std::vector<ge::Tensor> outputs;
    ret = session->RunGraph(0, inputs, outputs);
    if (ret != ge::SUCCESS || outputs.size() != names.size()) {
      std::cerr << "RunGraph failed step=" << step << " ret=" << ret
                << " outputs=" << outputs.size() << std::endl;
      ge::GEFinalize();
      return 7;
    }
    for (size_t index = 0; index < outputs.size(); ++index) {
      if (!WriteTensor(StepPath(argv[4], step, names[index]), outputs[index], sizes[index])) {
        ge::GEFinalize();
        return 8;
      }
    }
    std::cout << "G4A_ATTEMPT52_STEP step=" << step << " outputs=" << outputs.size()
              << std::endl;
  }
  ge::GEFinalize();
  std::cout << "G4A_ATTEMPT52_NATIVE_COMPLETE" << std::endl;
  return 0;
}
