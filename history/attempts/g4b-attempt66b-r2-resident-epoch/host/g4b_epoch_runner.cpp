#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "all_ops.h"
#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "graph/graph.h"

namespace {
constexpr int32_t kVocabSize = 152064;
constexpr int32_t kMaxEpochSteps = 8;
constexpr int32_t kBlockSize = 128;
constexpr int32_t kPhysicalBlocks = 2;
constexpr int32_t kConfiguredEos = 151645;
constexpr size_t kCacheBytes =
    28ULL * 2ULL * 128ULL * 4ULL * 128ULL * sizeof(uint16_t);
constexpr size_t kLogitsBytes = static_cast<size_t>(kVocabSize) * sizeof(float);
constexpr size_t kLogitsHistoryBytes = kMaxEpochSteps * kLogitsBytes;
constexpr int32_t kFeedTimeoutMs = 600000;
constexpr int32_t kFetchTimeoutMs = 1800000;

enum FinishReason : int32_t {
  kFinishEos = 1,
  kFinishMaxSteps = 2,
};

struct InputSpec {
  const char *name;
  std::vector<int64_t> shape;
  ge::DataType dtype;
  size_t bytes;
};

const std::array<InputSpec, 8> kInputSpecs = {{
    {"token_id.bin", {1, 1}, ge::DT_INT64, sizeof(int64_t)},
    {"position.bin", {1}, ge::DT_INT64, sizeof(int64_t)},
    {"sequence_length.bin", {1, 1}, ge::DT_INT32, sizeof(int32_t)},
    {"key_cache.bin", {28, 2, 128, 4, 128}, ge::DT_BF16, kCacheBytes},
    {"slot_mapping.bin", {1}, ge::DT_INT32, sizeof(int32_t)},
    {"block_table.bin", {1, 2}, ge::DT_INT32, 2 * sizeof(int32_t)},
    {"value_cache.bin", {28, 2, 128, 4, 128}, ge::DT_BF16, kCacheBytes},
    {"explicit_tiling.bin", {72}, ge::DT_UINT8, 72},
}};

bool ReadFile(const std::string &path, size_t expected,
              std::vector<uint8_t> &data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const auto size = static_cast<size_t>(stream.tellg());
  if (size != expected) {
    std::cerr << "SIZE_MISMATCH path=" << path << " actual=" << size
              << " expected=" << expected << std::endl;
    return false;
  }
  data.resize(size);
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(data.data()),
              static_cast<std::streamsize>(size));
  return static_cast<bool>(stream);
}

bool LoadInputs(const std::string &input_dir,
                std::array<std::vector<uint8_t>, 8> &buffers) {
  for (size_t index = 0; index < kInputSpecs.size(); ++index) {
    const auto &spec = kInputSpecs[index];
    if (!ReadFile(input_dir + "/" + spec.name, spec.bytes, buffers[index])) {
      return false;
    }
  }
  return true;
}

ge::Tensor MakeTensor(std::vector<uint8_t> &data,
                      const std::vector<int64_t> &shape,
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

bool WriteRaw(const std::string &path, const void *data, size_t bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream || data == nullptr) return false;
  stream.write(reinterpret_cast<const char *>(data),
               static_cast<std::streamsize>(bytes));
  return static_cast<bool>(stream);
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
  return WriteRaw(path, tensor.GetData(), expected_bytes);
}

bool CopyTensor(const ge::Tensor &tensor, size_t expected_bytes,
                ge::DataType expected_dtype,
                std::vector<uint8_t> &destination) {
  if (tensor.GetData() == nullptr || tensor.GetSize() != expected_bytes ||
      tensor.GetTensorDesc().GetDataType() != expected_dtype) {
    return false;
  }
  const auto *begin = static_cast<const uint8_t *>(tensor.GetData());
  destination.assign(begin, begin + expected_bytes);
  return true;
}

bool ArgmaxFinite(const ge::Tensor &logits, int64_t &token) {
  if (logits.GetData() == nullptr || logits.GetSize() != kLogitsBytes ||
      logits.GetTensorDesc().GetDataType() != ge::DT_FLOAT) {
    return false;
  }
  const auto *values = reinterpret_cast<const float *>(logits.GetData());
  if (!std::isfinite(values[0])) return false;
  float best = values[0];
  token = 0;
  for (int64_t index = 1; index < kVocabSize; ++index) {
    if (!std::isfinite(values[index])) return false;
    if (values[index] > best) {
      best = values[index];
      token = index;
    }
  }
  return true;
}

int32_t ComputeSlot(const int32_t *block_table, int64_t position) {
  if (position < 0 || position >= kMaxEpochSteps) return -1;
  const int32_t logical_block = static_cast<int32_t>(position / kBlockSize);
  const int32_t offset = static_cast<int32_t>(position % kBlockSize);
  const int32_t physical_block = block_table[logical_block];
  if (physical_block < 0 || physical_block >= kPhysicalBlocks) return -1;
  return physical_block * kBlockSize + offset;
}

bool WriteRuntime(const std::string &path, int64_t host_submissions,
                  int64_t feed_calls, int64_t fetch_calls, int64_t wall_us,
                  int64_t cpu_us) {
  std::ofstream stream(path, std::ios::trunc);
  if (!stream) return false;
  stream << "metric\tvalue\n"
         << "host_model_submissions\t" << host_submissions << '\n'
         << "feed_calls\t" << feed_calls << '\n'
         << "fetch_calls\t" << fetch_calls << '\n'
         << "wall_us\t" << wall_us << '\n'
         << "process_cpu_us\t" << cpu_us << '\n';
  return static_cast<bool>(stream);
}

bool SaveHostCase(const std::string &output_dir,
                  const std::vector<float> &logits_history,
                  const std::array<int64_t, kMaxEpochSteps> &token_history,
                  const std::vector<uint8_t> &key_cache,
                  const std::vector<uint8_t> &value_cache,
                  const std::vector<uint8_t> &position,
                  const std::array<int32_t, 12> &control,
                  int64_t host_submissions, int64_t wall_us, int64_t cpu_us) {
  return WriteRaw(output_dir + "/logits_history.bin", logits_history.data(),
                  kLogitsHistoryBytes) &&
         WriteRaw(output_dir + "/token_history.bin", token_history.data(),
                  token_history.size() * sizeof(int64_t)) &&
         WriteRaw(output_dir + "/key_cache.bin", key_cache.data(),
                  key_cache.size()) &&
         WriteRaw(output_dir + "/value_cache.bin", value_cache.data(),
                  value_cache.size()) &&
         WriteRaw(output_dir + "/final_position.bin", position.data(),
                  position.size()) &&
         WriteRaw(output_dir + "/control.bin", control.data(),
                  control.size() * sizeof(int32_t)) &&
         WriteRuntime(output_dir + "/runtime.tsv", host_submissions, 0, 0,
                      wall_us, cpu_us);
}

int RunHostCase(const std::shared_ptr<ge::Session> &session,
                const std::array<std::vector<uint8_t>, 8> &base,
                const std::string &output_dir, int32_t max_steps,
                int32_t eos_token, int64_t &first_token) {
  auto token = base[0];
  auto position = base[1];
  auto sequence_length = base[2];
  auto key_cache = base[3];
  auto slot_mapping = base[4];
  auto block_table = base[5];
  auto value_cache = base[6];
  auto tiling = base[7];
  const auto *blocks = reinterpret_cast<const int32_t *>(block_table.data());
  std::vector<float> logits_history(
      static_cast<size_t>(kMaxEpochSteps) * kVocabSize, 0.0F);
  std::array<int64_t, kMaxEpochSteps> token_history;
  token_history.fill(-1);
  int32_t executed = 0;
  int32_t finish_reason = kFinishMaxSteps;
  int64_t generated_token = -1;
  const auto wall_start = std::chrono::steady_clock::now();
  const auto cpu_start = std::clock();

  while (executed < max_steps) {
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
    const auto ret = session->RunGraph(0, inputs, outputs);
    if (ret != ge::SUCCESS || outputs.size() != 4 ||
        !ArgmaxFinite(outputs[0], generated_token)) {
      std::cerr << "HOST_RUN_FAILED step=" << executed
                << " ret=" << ret << " outputs=" << outputs.size()
                << std::endl;
      return 20;
    }
    std::memcpy(logits_history.data() +
                    static_cast<size_t>(executed) * kVocabSize,
                outputs[0].GetData(), kLogitsBytes);
    token_history[executed] = generated_token;
    if (executed == 0) first_token = generated_token;
    if (!CopyTensor(outputs[1], kCacheBytes, ge::DT_BF16, key_cache) ||
        !CopyTensor(outputs[2], kCacheBytes, ge::DT_BF16, value_cache) ||
        !CopyTensor(outputs[3], sizeof(int64_t), ge::DT_INT64, position)) {
      std::cerr << "HOST_OUTPUT_COPY_FAILED step=" << executed << std::endl;
      return 21;
    }
    ++executed;
    int64_t next_position = -1;
    std::memcpy(&next_position, position.data(), sizeof(next_position));
    if (next_position != executed) {
      std::cerr << "HOST_POSITION_MISMATCH expected=" << executed
                << " actual=" << next_position << std::endl;
      return 22;
    }
    if (generated_token == eos_token) {
      finish_reason = kFinishEos;
      break;
    }
    if (executed >= max_steps) {
      finish_reason = kFinishMaxSteps;
      break;
    }
    const int32_t next_slot = ComputeSlot(blocks, next_position);
    if (next_slot < 0) return 23;
    token = ScalarBytes<int64_t>(generated_token);
    sequence_length = ScalarBytes<int32_t>(
        static_cast<int32_t>(next_position + 1));
    slot_mapping = ScalarBytes<int32_t>(next_slot);
  }

  const auto cpu_end = std::clock();
  const auto wall_end = std::chrono::steady_clock::now();
  int64_t final_position = -1;
  std::memcpy(&final_position, position.data(), sizeof(final_position));
  std::array<int32_t, 12> control = {{
      max_steps,
      eos_token,
      0,
      0,
      executed,
      finish_reason,
      0,
      static_cast<int32_t>(generated_token),
      static_cast<int32_t>(final_position),
      static_cast<int32_t>(final_position + 1),
      executed,
      0,
  }};
  const int64_t wall_us =
      std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start)
          .count();
  const int64_t cpu_us = static_cast<int64_t>(
      1000000.0 * static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC);
  if (!SaveHostCase(output_dir, logits_history, token_history, key_cache,
                    value_cache, position, control, executed, wall_us,
                    cpu_us)) {
    return 24;
  }
  std::cout << "HOST_EPOCH_CASE max_steps=" << max_steps
            << " eos=" << eos_token << " executed=" << executed
            << " finish=" << finish_reason << " final_token="
            << generated_token << " wall_us=" << wall_us << std::endl;
  return 0;
}

int RunHostSuite(const std::string &air_path, const std::string &input_dir,
                 const std::string &output_dir) {
  std::array<std::vector<uint8_t>, 8> base;
  if (!LoadInputs(input_dir, base)) return 10;
  ge::Graph graph("Attempt65HostGenerationEpoch");
  const auto load_status = graph.LoadFromFile(air_path.c_str());
  std::cout << "HOST_AIR_LOAD status=" << load_status
            << " valid=" << graph.IsValid() << std::endl;
  if (load_status != ge::GRAPH_SUCCESS || !graph.IsValid()) return 11;
  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
  auto ret = ge::GEInitialize(options);
  if (ret != ge::SUCCESS) return 12;
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, graph);
  if (ret != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return 13;
  }
  const std::array<int32_t, 4> step_counts = {{1, 2, 4, 8}};
  int64_t first_token = -1;
  for (const auto steps : step_counts) {
    const auto status = RunHostCase(session, base,
                                    output_dir + "/k" +
                                        std::to_string(steps),
                                    steps, kConfiguredEos, first_token);
    if (status != 0) {
      session.reset();
      ge::GEFinalize();
      return status;
    }
  }
  if (first_token < 0 || first_token >= kVocabSize) {
    session.reset();
    ge::GEFinalize();
    return 14;
  }
  {
    std::ofstream token_file(output_dir + "/early_eos_token.txt",
                             std::ios::trunc);
    token_file << first_token << '\n';
  }
  const auto early_status = RunHostCase(session, base, output_dir + "/early-eos",
                                        8, static_cast<int32_t>(first_token),
                                        first_token);
  session.reset();
  ge::GEFinalize();
  if (early_status != 0) return early_status;
  std::cout << "HOST_EPOCH_SUITE_COMPLETE early_eos_token=" << first_token
            << std::endl;
  return 0;
}

ge::dflow::FlowGraph BuildDeviceFlow(const std::string &air_path,
                                     const std::string &graph_config,
                                     const std::string &func_config) {
  auto data0 = ge::dflow::FlowData("input0", 0);
  auto data1 = ge::dflow::FlowData("input1", 1);
  auto data2 = ge::dflow::FlowData("input2", 2);
  auto data3 = ge::dflow::FlowData("input3", 3);
  auto data4 = ge::dflow::FlowData("input4", 4);
  auto data5 = ge::dflow::FlowData("input5", 5);
  auto data6 = ge::dflow::FlowData("input6", 6);
  auto data7 = ge::dflow::FlowData("input7", 7);
  auto control = ge::dflow::FlowData("control", 8);
  auto graph_pp = ge::dflow::GraphPp(
      "attempt65_decoder_graph_pp", [air_path]() {
        ge::Graph graph("Attempt65InvokedDecoder");
        const auto status = graph.LoadFromFile(air_path.c_str());
        std::cout << "DEVICE_AIR_LOAD status=" << status
                  << " valid=" << graph.IsValid() << std::endl;
        return graph;
      });
  graph_pp.SetCompileConfig(graph_config.c_str());
  auto function_pp = ge::dflow::FunctionPp("g4b_resident_epoch_pp")
                         .SetCompileConfig(func_config.c_str());
  function_pp.AddInvokedClosure("decode_graph_0", graph_pp);
  auto node = ge::dflow::FlowNode("g4b_resident_epoch_node", 9, 6);
  node.AddPp(function_pp)
      .SetInput(0, data0)
      .SetInput(1, data1)
      .SetInput(2, data2)
      .SetInput(3, data3)
      .SetInput(4, data4)
      .SetInput(5, data5)
      .SetInput(6, data6)
      .SetInput(7, data7)
      .SetInput(8, control);
  ge::dflow::FlowGraph flow_graph("attempt66b_r2_resident_epoch");
  std::vector<ge::dflow::FlowOperator> inputs = {
      data0, data1, data2, data3, data4, data5, data6, data7, control};
  std::vector<std::pair<ge::dflow::FlowOperator, std::vector<size_t>>> outputs = {
      {node, {0, 1, 2, 3, 4, 5}}};
  flow_graph.SetInputs(inputs).SetOutputs(outputs);
  return flow_graph;
}

int RunDeviceCase(const std::shared_ptr<ge::Session> &session,
                  const std::array<std::vector<uint8_t>, 8> &base,
                  const std::string &output_dir, int32_t max_steps,
                  int32_t eos_token) {
  auto buffers = base;
  std::array<int32_t, 4> control_values = {{max_steps, eos_token, 0, 0}};
  std::vector<uint8_t> control_bytes(sizeof(control_values));
  std::memcpy(control_bytes.data(), control_values.data(), control_bytes.size());
  std::vector<ge::Tensor> inputs;
  inputs.reserve(9);
  for (size_t index = 0; index < kInputSpecs.size(); ++index) {
    inputs.push_back(MakeTensor(buffers[index], kInputSpecs[index].shape,
                                kInputSpecs[index].dtype));
  }
  inputs.push_back(MakeTensor(control_bytes, {4}, ge::DT_INT32));
  ge::DataFlowInfo flow_info;
  const auto wall_start = std::chrono::steady_clock::now();
  const auto cpu_start = std::clock();
  auto ret = session->FeedDataFlowGraph(0, inputs, flow_info, kFeedTimeoutMs);
  const auto feed_end = std::chrono::steady_clock::now();
  if (ret != ge::SUCCESS) {
    std::cerr << "DEVICE_FEED_FAILED max_steps=" << max_steps
              << " ret=" << ret << std::endl;
    return 30;
  }
  std::vector<ge::Tensor> outputs;
  ret = session->FetchDataFlowGraph(0, outputs, flow_info, kFetchTimeoutMs);
  const auto wall_end = std::chrono::steady_clock::now();
  const auto cpu_end = std::clock();
  if (ret != ge::SUCCESS || outputs.size() != 6) {
    std::cerr << "DEVICE_FETCH_FAILED max_steps=" << max_steps
              << " ret=" << ret << " outputs=" << outputs.size()
              << std::endl;
    return 31;
  }
  const std::array<const char *, 6> names = {{
      "logits_history.bin", "token_history.bin", "key_cache.bin",
      "value_cache.bin", "final_position.bin", "control.bin"}};
  const std::array<size_t, 6> sizes = {{
      kLogitsHistoryBytes,
      kMaxEpochSteps * sizeof(int64_t),
      kCacheBytes,
      kCacheBytes,
      sizeof(int64_t),
      12 * sizeof(int32_t),
  }};
  const std::array<ge::DataType, 6> dtypes = {{
      ge::DT_FLOAT, ge::DT_INT64, ge::DT_BF16,
      ge::DT_BF16, ge::DT_INT64, ge::DT_INT32,
  }};
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (!WriteTensor(output_dir + "/" + names[index], outputs[index],
                     sizes[index], dtypes[index])) {
      return 32;
    }
  }
  const int64_t wall_us =
      std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start)
          .count();
  const int64_t cpu_us = static_cast<int64_t>(
      1000000.0 * static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC);
  const int64_t feed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(feed_end - wall_start)
          .count();
  if (!WriteRuntime(output_dir + "/runtime.tsv", 1, 1, 1, wall_us, cpu_us)) {
    return 33;
  }
  std::cout << "DEVICE_EPOCH_CASE max_steps=" << max_steps
            << " eos=" << eos_token << " feed_us=" << feed_us
            << " total_us=" << wall_us
            << " feed_calls=1 fetch_calls=1" << std::endl;
  return 0;
}

int RunDeviceSuite(const std::string &air_path,
                   const std::string &graph_config,
                   const std::string &func_config,
                   const std::string &input_dir,
                   const std::string &output_dir, int32_t early_eos_token) {
  std::array<std::vector<uint8_t>, 8> base;
  if (!LoadInputs(input_dir, base)) return 40;
  auto flow_graph = BuildDeviceFlow(air_path, graph_config, func_config);
  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
      {"ge.graphRunMode", "0"}};
  auto ret = ge::GEInitialize(options);
  if (ret != ge::SUCCESS) return 41;
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, flow_graph.ToGeGraph());
  if (ret != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return 42;
  }
  const std::array<int32_t, 4> step_counts = {{1, 2, 4, 8}};
  for (const auto steps : step_counts) {
    const auto status = RunDeviceCase(session, base,
                                      output_dir + "/k" +
                                          std::to_string(steps),
                                      steps, kConfiguredEos);
    if (status != 0) {
      session.reset();
      ge::GEFinalize();
      return status;
    }
  }
  const auto early_status =
      RunDeviceCase(session, base, output_dir + "/early-eos", 8,
                    early_eos_token);
  session.reset();
  ge::GEFinalize();
  if (early_status != 0) return early_status;
  std::cout << "DEVICE_EPOCH_SUITE_COMPLETE early_eos_token="
            << early_eos_token << std::endl;
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: g4b_epoch_runner ROUTE AIR GRAPH_CONFIG FUNC_CONFIG "
                 "INPUT_DIR OUTPUT_DIR EARLY_EOS_TOKEN\n";
    return 2;
  }
  const std::string route = argv[1];
  if (route == "host") {
    return RunHostSuite(argv[2], argv[5], argv[6]);
  }
  if (route == "device") {
    const auto early_eos = static_cast<int32_t>(std::stol(argv[7]));
    if (early_eos < 0 || early_eos >= kVocabSize) return 3;
    return RunDeviceSuite(argv[2], argv[3], argv[4], argv[5], argv[6],
                          early_eos);
  }
  std::cerr << "unsupported route: " << route << std::endl;
  return 2;
}
