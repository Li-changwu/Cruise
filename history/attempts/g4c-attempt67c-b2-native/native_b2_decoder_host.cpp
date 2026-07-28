#include <array>
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
constexpr size_t kTokenBytes = 2ULL * sizeof(int64_t);
constexpr size_t kPositionBytes = 2ULL * sizeof(int64_t);
constexpr size_t kSequenceLengthBytes = 2ULL * sizeof(int32_t);
constexpr size_t kCacheBytes =
    28ULL * 4ULL * 128ULL * 4ULL * 128ULL * sizeof(uint16_t);
constexpr size_t kSlotMappingBytes = 2ULL * sizeof(int32_t);
constexpr size_t kActiveMaskBytes = 2ULL * sizeof(int32_t);
constexpr size_t kBlockTableBytes = 4ULL * sizeof(int32_t);
constexpr size_t kTilingBytes = 72ULL;
constexpr size_t kLogitsBytes = 2ULL * 152064ULL * sizeof(float);

struct InputSpec {
  const char *filename;
  std::vector<int64_t> shape;
  ge::DataType dtype;
  size_t bytes;
};

const std::array<InputSpec, 9> kInputs = {{
    {"token_id.bin", {2, 1}, ge::DT_INT64, kTokenBytes},
    {"position.bin", {2}, ge::DT_INT64, kPositionBytes},
    {"sequence_length.bin", {2, 1}, ge::DT_INT32, kSequenceLengthBytes},
    {"key_cache.bin", {28, 4, 128, 4, 128}, ge::DT_BF16, kCacheBytes},
    {"slot_mapping.bin", {2}, ge::DT_INT32, kSlotMappingBytes},
    {"active_mask.bin", {2}, ge::DT_INT32, kActiveMaskBytes},
    {"block_table.bin", {2, 2}, ge::DT_INT32, kBlockTableBytes},
    {"value_cache.bin", {28, 4, 128, 4, 128}, ge::DT_BF16, kCacheBytes},
    {"explicit_tiling.bin", {72}, ge::DT_UINT8, kTilingBytes},
}};

bool ReadFile(const std::string &path, size_t expected,
              std::vector<uint8_t> &data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const auto size = static_cast<size_t>(stream.tellg());
  if (size != expected) {
    std::cerr << "INPUT_SIZE_MISMATCH path=" << path << " actual=" << size
              << " expected=" << expected << std::endl;
    return false;
  }
  data.resize(size);
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(data.data()),
              static_cast<std::streamsize>(size));
  return static_cast<bool>(stream);
}

ge::Tensor MakeTensor(std::vector<uint8_t> &data,
                      const std::vector<int64_t> &shape,
                      ge::DataType dtype) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(
      ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype));
  tensor.SetData(data.data(), data.size());
  return tensor;
}

bool WriteTensor(const std::string &path, const ge::Tensor &tensor,
                 size_t expected_bytes, ge::DataType expected_dtype) {
  if (tensor.GetData() == nullptr || tensor.GetSize() != expected_bytes ||
      tensor.GetTensorDesc().GetDataType() != expected_dtype) {
    std::cerr << "OUTPUT_CONTRACT_MISMATCH path=" << path
              << " bytes=" << tensor.GetSize()
              << " dtype="
              << static_cast<int>(tensor.GetTensorDesc().GetDataType())
              << std::endl;
    return false;
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return false;
  stream.write(reinterpret_cast<const char *>(tensor.GetData()),
               static_cast<std::streamsize>(tensor.GetSize()));
  return static_cast<bool>(stream);
}

int RunCase(const std::shared_ptr<ge::Session> &session,
            const std::string &input_root, const std::string &output_root,
            int case_index) {
  const std::string prefix = "case" + std::to_string(case_index);
  const std::string input_dir = input_root + "/" + prefix;
  std::array<std::vector<uint8_t>, 9> buffers;
  std::vector<ge::Tensor> inputs;
  inputs.reserve(kInputs.size());
  for (size_t index = 0; index < kInputs.size(); ++index) {
    const auto &spec = kInputs[index];
    if (!ReadFile(input_dir + "/" + spec.filename, spec.bytes,
                  buffers[index])) {
      return 10;
    }
    inputs.push_back(MakeTensor(buffers[index], spec.shape, spec.dtype));
  }
  std::vector<ge::Tensor> outputs;
  const auto ret = session->RunGraph(0, inputs, outputs);
  if (ret != ge::SUCCESS || outputs.size() != 4) {
    std::cerr << "RUN_GRAPH_FAILED case=" << case_index << " ret=" << ret
              << " outputs=" << outputs.size() << std::endl;
    return 11;
  }
  const std::array<const char *, 4> names = {{
      "logits.bin", "key_cache.bin", "value_cache.bin", "next_position.bin"}};
  const std::array<size_t, 4> sizes = {{
      kLogitsBytes, kCacheBytes, kCacheBytes, kPositionBytes}};
  const std::array<ge::DataType, 4> dtypes = {{
      ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_INT64}};
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (!WriteTensor(output_root + "/" + prefix + "_" + names[index],
                     outputs[index], sizes[index], dtypes[index])) {
      return 12;
    }
  }
  std::cout << "G4C_ATTEMPT67C_CASE case=" << case_index
            << " outputs=" << outputs.size() << std::endl;
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: native_b2_decoder_host AIR INPUT_ROOT OUTPUT_ROOT\n";
    return 2;
  }
  ge::Graph graph("G4cAttempt67cB2Decoder");
  const auto load_status = graph.LoadFromFile(argv[1]);
  std::cout << "G4C_ATTEMPT67C_LOAD status=" << load_status
            << " valid=" << graph.IsValid() << std::endl;
  if (load_status != ge::GRAPH_SUCCESS || !graph.IsValid()) return 3;
  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) return static_cast<int>(ret);
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, graph);
  if (ret != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return 4;
  }
  for (int case_index = 0; case_index < 3; ++case_index) {
    const auto status = RunCase(session, argv[2], argv[3], case_index);
    if (status != 0) {
      session.reset();
      ge::GEFinalize();
      return status;
    }
  }
  session.reset();
  ge::GEFinalize();
  std::cout << "G4C_ATTEMPT67C_NATIVE_COMPLETE" << std::endl;
  return 0;
}

