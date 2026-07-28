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
constexpr size_t kTokenBytes = sizeof(int64_t);
constexpr size_t kPositionBytes = sizeof(int64_t);
constexpr size_t kSequenceLengthBytes = sizeof(int32_t);
constexpr size_t kCacheBytes = 28ULL * 2ULL * 128ULL * 4ULL * 128ULL * sizeof(uint16_t);
constexpr size_t kSlotMappingBytes = sizeof(int32_t);
constexpr size_t kBlockTableBytes = 2ULL * sizeof(int32_t);
constexpr size_t kTilingBytes = 72ULL;
constexpr size_t kLogitsBytes = 152064ULL * sizeof(float);

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
  if (argc != 5) {
    std::cerr << "usage: native_full_decoder_attempt53k_host AIR INPUT_DIR OUTPUT_DIR CACHE_DIR\n";
    return 2;
  }

  ge::Graph graph("G4aAttempt53kCompleteDecoder");
  const auto load_status = graph.LoadFromFile(argv[1]);
  std::cout << "G4A_ATTEMPT53K_LOAD status=" << load_status
            << " valid=" << graph.IsValid() << std::endl;
  if (load_status != ge::GRAPH_SUCCESS || !graph.IsValid()) return 3;

  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"}};
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
      std::cerr << "position recurrence mismatch step=" << step
                << " position=" << current_position << std::endl;
      ge::GEFinalize();
      return 6;
    }

    std::vector<uint8_t> token;
    if (!ReadFile(StepPath(input_dir, step, "token_id"), kTokenBytes, token)) {
      ge::GEFinalize();
      return 7;
    }
    auto sequence_length = ScalarBytes<int32_t>(static_cast<int32_t>(current_position + 1));
    auto slot_mapping = ScalarBytes<int32_t>(static_cast<int32_t>(128 + current_position));
    std::vector<ge::Tensor> inputs = {
        MakeTensor(token, {1, 1}, ge::DT_INT64),
        MakeTensor(position, {1}, ge::DT_INT64),
        MakeTensor(sequence_length, {1, 1}, ge::DT_INT32),
        MakeTensor(key_cache, {28, 2, 128, 4, 128}, ge::DT_BF16),
        MakeTensor(slot_mapping, {1}, ge::DT_INT32),
        MakeTensor(block_table, {1, 2}, ge::DT_INT32),
        MakeTensor(value_cache, {28, 2, 128, 4, 128}, ge::DT_BF16),
        MakeTensor(tiling, {72}, ge::DT_UINT8)};
    std::vector<ge::Tensor> outputs;
    ret = session->RunGraph(0, inputs, outputs);
    if (ret != ge::SUCCESS || outputs.size() != 4) {
      std::cerr << "RunGraph failed step=" << step << " ret=" << ret
                << " outputs=" << outputs.size() << std::endl;
      ge::GEFinalize();
      return 8;
    }

    const std::vector<std::string> names = {"logits", "key_cache", "value_cache", "next_position"};
    const std::vector<size_t> sizes = {kLogitsBytes, kCacheBytes, kCacheBytes, kPositionBytes};
    for (size_t index = 0; index < outputs.size(); ++index) {
      if (!WriteTensor(StepPath(output_dir, step, names[index]), outputs[index], sizes[index])) {
        ge::GEFinalize();
        return 9;
      }
    }
    if (!CopyTensor(outputs[1], kCacheBytes, key_cache) ||
        !CopyTensor(outputs[2], kCacheBytes, value_cache) ||
        !CopyTensor(outputs[3], kPositionBytes, position)) {
      ge::GEFinalize();
      return 10;
    }
    std::cout << "G4A_ATTEMPT53K_STEP step=" << step << " outputs=" << outputs.size()
              << std::endl;
  }

  ge::GEFinalize();
  std::cout << "G4A_ATTEMPT53K_NATIVE_COMPLETE" << std::endl;
  return 0;
}
