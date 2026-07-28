#include <cstdint>
#include <cstring>
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
constexpr size_t kHiddenBytes = 1ULL * 1ULL * 3584ULL * sizeof(uint16_t);
constexpr size_t kCacheBytes = 2ULL * 128ULL * 4ULL * 128ULL * sizeof(uint16_t);
constexpr size_t kBlockTableBytes = 2ULL * sizeof(int32_t);
constexpr size_t kTilingBytes = 72ULL;

const std::vector<std::string> kOutputNames = {
    "updated_key", "updated_value", "next_position", "input_norm", "query_rope",
    "key_rope", "value_projection", "masked_scores", "probabilities",
    "attention_value", "attention_projection", "hidden_after_attention",
    "post_attention_norm", "gate_preactivation", "gate", "up", "mlp_product", "mlp_projection",
    "hidden_after_mlp"};
const std::vector<size_t> kOutputBytes = {
    kCacheBytes, kCacheBytes, sizeof(int64_t), kHiddenBytes,
    1ULL * 28ULL * 1ULL * 128ULL * sizeof(uint16_t),
    1ULL * 4ULL * 1ULL * 128ULL * sizeof(uint16_t),
    1ULL * 4ULL * 1ULL * 128ULL * sizeof(uint16_t),
    1ULL * 28ULL * 1ULL * 8ULL * sizeof(uint16_t),
    1ULL * 28ULL * 1ULL * 8ULL * sizeof(uint16_t),
    1ULL * 28ULL * 1ULL * 128ULL * sizeof(uint16_t),
    kHiddenBytes, kHiddenBytes, kHiddenBytes,
    1ULL * 1ULL * 18944ULL * sizeof(uint16_t),
    1ULL * 1ULL * 18944ULL * sizeof(uint16_t),
    1ULL * 1ULL * 18944ULL * sizeof(uint16_t),
    1ULL * 1ULL * 18944ULL * sizeof(uint16_t),
    kHiddenBytes, kHiddenBytes};

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

template <typename T>
std::vector<uint8_t> ScalarBytes(T value) {
  std::vector<uint8_t> data(sizeof(T));
  std::memcpy(data.data(), &value, sizeof(T));
  return data;
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

bool CopyTensor(const ge::Tensor &tensor, size_t expected, std::vector<uint8_t> &destination) {
  if (tensor.GetData() == nullptr || tensor.GetSize() != expected) return false;
  const auto *begin = reinterpret_cast<const uint8_t *>(tensor.GetData());
  destination.assign(begin, begin + expected);
  return true;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: native_layer0_attempt59a_host AIR INPUT_DIR OUTPUT_DIR\n";
    return 2;
  }
  ge::Graph graph("G4aAttempt59aGateMatMulV2TransposeX2");
  const auto load_status = graph.LoadFromFile(argv[1]);
  std::cout << "G4A_ATTEMPT59A_LOAD status=" << load_status
            << " valid=" << graph.IsValid() << std::endl;
  if (load_status != ge::GRAPH_SUCCESS || !graph.IsValid()) return 3;

  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) return static_cast<int>(ret);
  auto session = std::make_shared<ge::Session>(std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, graph);
  if (ret != ge::SUCCESS) {
    ge::GEFinalize();
    return 4;
  }

  const std::string input_dir = argv[2];
  const std::string output_dir = argv[3];
  std::vector<uint8_t> key_cache, value_cache, block_table, tiling;
  if (!ReadFile(input_dir + "/initial_key_cache.bin", kCacheBytes, key_cache) ||
      !ReadFile(input_dir + "/initial_value_cache.bin", kCacheBytes, value_cache) ||
      !ReadFile(input_dir + "/block_table.bin", kBlockTableBytes, block_table) ||
      !ReadFile(input_dir + "/tiling.bin", kTilingBytes, tiling)) {
    ge::GEFinalize();
    return 5;
  }

  std::vector<uint8_t> position = ScalarBytes<int64_t>(0);
  for (int step = 1; step <= 4; ++step) {
    int64_t current_position = -1;
    std::memcpy(&current_position, position.data(), sizeof(current_position));
    if (current_position != step - 1) {
      ge::GEFinalize();
      return 6;
    }
    std::vector<uint8_t> hidden;
    if (!ReadFile(StepPath(input_dir, step, "hidden"), kHiddenBytes, hidden)) {
      ge::GEFinalize();
      return 7;
    }
    auto slot_mapping = ScalarBytes<int32_t>(static_cast<int32_t>(128 + current_position));
    auto sequence_length = ScalarBytes<int32_t>(static_cast<int32_t>(current_position + 1));
    std::vector<ge::Tensor> inputs = {
        MakeTensor(hidden, {1, 1, 3584}, ge::DT_BF16),
        MakeTensor(position, {1}, ge::DT_INT64),
        MakeTensor(key_cache, {2, 128, 4, 128}, ge::DT_BF16),
        MakeTensor(slot_mapping, {1}, ge::DT_INT32),
        MakeTensor(block_table, {1, 2}, ge::DT_INT32),
        MakeTensor(value_cache, {2, 128, 4, 128}, ge::DT_BF16),
        MakeTensor(tiling, {72}, ge::DT_UINT8),
        MakeTensor(sequence_length, {1, 1}, ge::DT_INT32)};
    std::vector<ge::Tensor> outputs;
    ret = session->RunGraph(0, inputs, outputs);
    if (ret != ge::SUCCESS || outputs.size() != kOutputNames.size()) {
      std::cerr << "RunGraph failed step=" << step << " ret=" << ret
                << " outputs=" << outputs.size() << std::endl;
      ge::GEFinalize();
      return 8;
    }
    for (size_t index = 0; index < outputs.size(); ++index) {
      if (!WriteTensor(StepPath(output_dir, step, kOutputNames[index]),
                       outputs[index], kOutputBytes[index])) {
        ge::GEFinalize();
        return 9;
      }
    }
    if (!CopyTensor(outputs[0], kCacheBytes, key_cache) ||
        !CopyTensor(outputs[1], kCacheBytes, value_cache) ||
        !CopyTensor(outputs[2], sizeof(int64_t), position)) {
      ge::GEFinalize();
      return 10;
    }
    std::cout << "G4A_ATTEMPT59A_STEP step=" << step
              << " outputs=" << outputs.size() << std::endl;
  }
  ge::GEFinalize();
  std::cout << "G4A_ATTEMPT59A_NATIVE_COMPLETE" << std::endl;
  return 0;
}
