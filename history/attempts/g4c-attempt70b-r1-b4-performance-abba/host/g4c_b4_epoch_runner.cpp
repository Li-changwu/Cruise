#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
constexpr int32_t kBatchSize = 4;
constexpr int32_t kVocabSize = 152064;
constexpr int32_t kMaxEpochSteps = 8;
constexpr int32_t kLogicalCapacity = 8;
constexpr int32_t kBlockSize = 128;
constexpr int32_t kBlocksPerRequest = 2;
constexpr int32_t kPhysicalBlocks = 8;
constexpr int32_t kConfiguredEos = 151645;
constexpr int32_t kControlInputElements = 1 + kBatchSize + 2;
constexpr int32_t kControlOutputElements = 6 + 4 * kBatchSize + 2;
constexpr int32_t kControlEosOffset = 6;
constexpr int32_t kControlInitialActiveOffset = kControlEosOffset + kBatchSize;
constexpr int32_t kControlExecutedOffset =
    kControlInitialActiveOffset + kBatchSize;
constexpr int32_t kControlReasonOffset = kControlExecutedOffset + kBatchSize;
constexpr int32_t kControlInitialCount = kControlReasonOffset + kBatchSize;
constexpr int32_t kControlFinalCount = kControlInitialCount + 1;
constexpr size_t kCacheBytes =
    28ULL * 8ULL * 128ULL * 4ULL * 128ULL * sizeof(uint16_t);
constexpr size_t kStepLogitsBytes =
    static_cast<size_t>(kBatchSize) * kVocabSize * sizeof(float);
constexpr size_t kLogitsHistoryBytes =
    kMaxEpochSteps * kStepLogitsBytes;
constexpr int32_t kFeedTimeoutMs = 600000;
constexpr int32_t kFetchTimeoutMs = 3600000;

enum FinishReason : int32_t {
  kFinishNone = 0,
  kFinishEos = 1,
  kFinishMaxSteps = 2,
  kFinishEmpty = 4,
  kFinishAlreadyFinished = 5,
};

struct InputSpec {
  const char *name;
  std::vector<int64_t> shape;
  ge::DataType dtype;
  size_t bytes;
};

struct Timing {
  int64_t wall_us = 0;
  int64_t cpu_us = 0;
};

const std::array<InputSpec, 9> kInputSpecs = {{
    {"token_id.bin", {4, 1}, ge::DT_INT64,
     kBatchSize * sizeof(int64_t)},
    {"position.bin", {4}, ge::DT_INT64, kBatchSize * sizeof(int64_t)},
    {"sequence_length.bin", {4, 1}, ge::DT_INT32,
     kBatchSize * sizeof(int32_t)},
    {"key_cache.bin", {28, 8, 128, 4, 128}, ge::DT_BF16, kCacheBytes},
    {"slot_mapping.bin", {4}, ge::DT_INT32,
     kBatchSize * sizeof(int32_t)},
    {"active_mask.bin", {4}, ge::DT_INT32,
     kBatchSize * sizeof(int32_t)},
    {"block_table.bin", {4, 2}, ge::DT_INT32,
     kBatchSize * kBlocksPerRequest * sizeof(int32_t)},
    {"value_cache.bin", {28, 8, 128, 4, 128}, ge::DT_BF16,
     kCacheBytes},
    {"explicit_tiling.bin", {72}, ge::DT_UINT8, 72},
}};

const std::array<const char *, 6> kRegularCases = {{
    "k1-heterogeneous", "k2-heterogeneous", "k4-heterogeneous",
    "k8-all-active", "active-empty-alternating",
    "finished-active-empty-active"}};
const std::array<int32_t, 6> kRegularSteps = {{1, 2, 4, 8, 4, 4}};
const char *kEarlyCase = "independent-early-eos";

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
                std::array<std::vector<uint8_t>, 9> &buffers) {
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

bool WriteRaw(const std::string &path, const void *data, size_t bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return false;
  stream.write(reinterpret_cast<const char *>(data),
               static_cast<std::streamsize>(bytes));
  return static_cast<bool>(stream);
}

bool WriteTensor(const std::string &path, const ge::Tensor &tensor,
                 size_t expected_bytes, ge::DataType expected_dtype) {
  if (tensor.GetData() == nullptr || tensor.GetSize() != expected_bytes ||
      tensor.GetTensorDesc().GetDataType() != expected_dtype) {
    std::cerr << "OUTPUT_MISMATCH path=" << path
              << " bytes=" << tensor.GetSize()
              << " dtype=" << tensor.GetTensorDesc().GetDataType()
              << std::endl;
    return false;
  }
  return WriteRaw(path, tensor.GetData(), expected_bytes);
}

bool CopyTensor(const ge::Tensor &tensor, size_t expected_bytes,
                ge::DataType expected_dtype, std::vector<uint8_t> &out) {
  if (tensor.GetData() == nullptr || tensor.GetSize() != expected_bytes ||
      tensor.GetTensorDesc().GetDataType() != expected_dtype) {
    return false;
  }
  const auto *begin = static_cast<const uint8_t *>(tensor.GetData());
  out.assign(begin, begin + expected_bytes);
  return true;
}

bool ArgmaxFinite(const ge::Tensor &logits, int32_t request, int64_t &token) {
  if (logits.GetData() == nullptr || logits.GetSize() != kStepLogitsBytes ||
      logits.GetTensorDesc().GetDataType() != ge::DT_FLOAT) {
    return false;
  }
  const auto *values = reinterpret_cast<const float *>(logits.GetData()) +
                       static_cast<int64_t>(request) * kVocabSize;
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

int32_t ComputeSlot(const int32_t *block_table, int32_t request,
                    int64_t position) {
  if (request < 0 || request >= kBatchSize || position < 0 ||
      position >= kLogicalCapacity) {
    return -1;
  }
  const int32_t logical_block = static_cast<int32_t>(position / kBlockSize);
  if (logical_block >= kBlocksPerRequest) return -1;
  const int32_t physical =
      block_table[request * kBlocksPerRequest + logical_block];
  if (physical < 0 || physical >= kPhysicalBlocks) return -1;
  return physical * kBlockSize + static_cast<int32_t>(position % kBlockSize);
}

int32_t CountActive(const int32_t *active) {
  int32_t count = 0;
  for (int32_t request = 0; request < kBatchSize; ++request) {
    if (active[request] != 0) ++count;
  }
  return count;
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
                  const std::array<int64_t,
                                   kMaxEpochSteps * kBatchSize> &token_history,
                  const std::array<std::vector<uint8_t>, 9> &state,
                  const std::array<int32_t, kControlOutputElements> &control,
                  int64_t model_calls, int64_t wall_us, int64_t cpu_us) {
  return WriteRaw(output_dir + "/logits_history.bin", logits_history.data(),
                  kLogitsHistoryBytes) &&
         WriteRaw(output_dir + "/token_history.bin", token_history.data(),
                  token_history.size() * sizeof(int64_t)) &&
         WriteRaw(output_dir + "/key_cache.bin", state[3].data(),
                  state[3].size()) &&
         WriteRaw(output_dir + "/value_cache.bin", state[7].data(),
                  state[7].size()) &&
         WriteRaw(output_dir + "/final_token.bin", state[0].data(),
                  state[0].size()) &&
         WriteRaw(output_dir + "/final_position.bin", state[1].data(),
                  state[1].size()) &&
         WriteRaw(output_dir + "/final_sequence_length.bin", state[2].data(),
                  state[2].size()) &&
         WriteRaw(output_dir + "/final_slot_mapping.bin", state[4].data(),
                  state[4].size()) &&
         WriteRaw(output_dir + "/final_active_mask.bin", state[5].data(),
                  state[5].size()) &&
         WriteRaw(output_dir + "/control.bin", control.data(),
                  control.size() * sizeof(int32_t)) &&
         WriteRuntime(output_dir + "/runtime.tsv", model_calls, 0, 0,
                      wall_us, cpu_us);
}

int RunHostCase(const std::shared_ptr<ge::Session> &session,
                 const std::array<std::vector<uint8_t>, 9> &base,
                 const std::string &output_dir, int32_t max_steps,
                 const std::array<int32_t, kBatchSize> &eos,
                 std::array<int64_t, kBatchSize> *first_tokens,
                 Timing *timing = nullptr, bool save_outputs = true) {
  auto state = base;
  auto *active = reinterpret_cast<int32_t *>(state[5].data());
  std::array<int32_t, kBatchSize> initial_active{};
  std::memcpy(initial_active.data(), active,
              static_cast<size_t>(kBatchSize) * sizeof(int32_t));
  const auto *initial_length =
      reinterpret_cast<const int32_t *>(state[2].data());
  std::array<int32_t, kBatchSize> executed{};
  std::array<int32_t, kBatchSize> reason{};
  for (int32_t request = 0; request < kBatchSize; ++request) {
    if (initial_active[request] == 0) {
      reason[request] = initial_length[request] == 0
                            ? kFinishEmpty
                            : kFinishAlreadyFinished;
    }
  }
  std::vector<float> logits_history(
      static_cast<size_t>(kMaxEpochSteps) * kBatchSize * kVocabSize, 0.0F);
  std::array<int64_t, kMaxEpochSteps * kBatchSize> token_history;
  token_history.fill(-1);
  int32_t model_calls = 0;
  const auto wall_start = std::chrono::steady_clock::now();
  const auto cpu_start = std::clock();

  while (model_calls < max_steps && CountActive(active) > 0) {
    std::array<int64_t, kBatchSize> previous_position;
    std::memcpy(previous_position.data(), state[1].data(), state[1].size());
    std::array<int32_t, kBatchSize> active_before{};
    std::memcpy(active_before.data(), active,
                static_cast<size_t>(kBatchSize) * sizeof(int32_t));
    std::vector<ge::Tensor> inputs;
    inputs.reserve(kInputSpecs.size());
    for (size_t index = 0; index < kInputSpecs.size(); ++index) {
      inputs.push_back(MakeTensor(state[index], kInputSpecs[index].shape,
                                  kInputSpecs[index].dtype));
    }
    std::vector<ge::Tensor> outputs;
    const auto ret = session->RunGraph(0, inputs, outputs);
    if (ret != ge::SUCCESS || outputs.size() != 4 ||
        outputs[0].GetSize() != kStepLogitsBytes) {
      std::cerr << "HOST_RUN_FAILED call=" << model_calls << " ret=" << ret
                << " outputs=" << outputs.size() << std::endl;
      return 20;
    }
    if (!CopyTensor(outputs[1], kCacheBytes, ge::DT_BF16, state[3]) ||
        !CopyTensor(outputs[2], kCacheBytes, ge::DT_BF16, state[7]) ||
        !CopyTensor(outputs[3], kBatchSize * sizeof(int64_t), ge::DT_INT64,
                    state[1])) {
      return 21;
    }
    const auto *next_position =
        reinterpret_cast<const int64_t *>(state[1].data());
    auto *token = reinterpret_cast<int64_t *>(state[0].data());
    auto *length = reinterpret_cast<int32_t *>(state[2].data());
    auto *slot = reinterpret_cast<int32_t *>(state[4].data());
    const auto *blocks = reinterpret_cast<const int32_t *>(state[6].data());
    const auto *step_logits =
        reinterpret_cast<const float *>(outputs[0].GetData());
    for (int32_t request = 0; request < kBatchSize; ++request) {
      if (active_before[request] == 0) {
        if (next_position[request] != previous_position[request]) return 22;
        continue;
      }
      int64_t generated = -1;
      if (!ArgmaxFinite(outputs[0], request, generated) ||
          next_position[request] != previous_position[request] + 1) {
        return 23;
      }
      const size_t history_offset =
          (static_cast<size_t>(model_calls) * kBatchSize + request) *
          kVocabSize;
      std::memcpy(logits_history.data() + history_offset,
                  step_logits + static_cast<size_t>(request) * kVocabSize,
                  static_cast<size_t>(kVocabSize) * sizeof(float));
      token_history[model_calls * kBatchSize + request] = generated;
      if (model_calls == 0 && first_tokens != nullptr) {
        (*first_tokens)[request] = generated;
      }
      ++executed[request];
      token[request] = generated;
      length[request] = static_cast<int32_t>(next_position[request] + 1);
      if (next_position[request] == kLogicalCapacity) {
        slot[request] = -1;
      } else {
        slot[request] = ComputeSlot(blocks, request, next_position[request]);
        if (slot[request] < 0) return 24;
      }
      if (generated == eos[request]) {
        active[request] = 0;
        reason[request] = kFinishEos;
      }
    }
    ++model_calls;
  }

  for (int32_t request = 0; request < kBatchSize; ++request) {
    if (initial_active[request] == 1 && reason[request] == kFinishNone) {
      reason[request] = kFinishMaxSteps;
    }
  }
  const auto cpu_end = std::clock();
  const auto wall_end = std::chrono::steady_clock::now();
  std::array<int32_t, kControlOutputElements> control{};
  control[0] = max_steps;
  control[1] = 0;
  control[2] = 0;
  control[3] = 0;
  control[4] = model_calls;
  control[5] = 0;
  for (int32_t request = 0; request < kBatchSize; ++request) {
    control[kControlEosOffset + request] = eos[request];
    control[kControlInitialActiveOffset + request] = initial_active[request];
    control[kControlExecutedOffset + request] = executed[request];
    control[kControlReasonOffset + request] = reason[request];
  }
  control[kControlInitialCount] = CountActive(initial_active.data());
  control[kControlFinalCount] = CountActive(active);
  const int64_t wall_us =
      std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start)
          .count();
  const int64_t cpu_us = static_cast<int64_t>(
      1000000.0 * static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC);
  if (timing != nullptr) {
    timing->wall_us = wall_us;
    timing->cpu_us = cpu_us;
  }
  if (save_outputs &&
      !SaveHostCase(output_dir, logits_history, token_history, state, control,
                    model_calls, wall_us, cpu_us)) {
    return 25;
  }
  std::cout << "HOST_B4_CASE name=" << output_dir
            << " calls=" << model_calls
            << " initial_active=" << CountActive(initial_active.data())
            << " final_active=" << CountActive(active)
            << " wall_us=" << wall_us << std::endl;
  return 0;
}

int RunHostSuite(const std::string &air_path, const std::string &input_root,
                 const std::string &output_root) {
  ge::Graph graph("Attempt69eB4HostEpoch");
  const auto load_status = graph.LoadFromFile(air_path.c_str());
  std::cout << "HOST_AIR_LOAD status=" << load_status
            << " valid=" << graph.IsValid() << std::endl;
  if (load_status != ge::GRAPH_SUCCESS || !graph.IsValid()) return 10;
  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
  auto ret = ge::GEInitialize(options);
  if (ret != ge::SUCCESS) return 11;
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, graph);
  if (ret != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return 12;
  }
  std::array<int32_t, kBatchSize> configured{};
  configured.fill(kConfiguredEos);
  std::array<int64_t, kBatchSize> first_tokens{};
  first_tokens.fill(-1);
  for (size_t index = 0; index < kRegularCases.size(); ++index) {
    std::array<std::vector<uint8_t>, 9> base;
    if (!LoadInputs(input_root + "/" + kRegularCases[index], base)) return 13;
    auto *capture = index == 0 ? &first_tokens : nullptr;
    const int status = RunHostCase(
        session, base, output_root + "/" + kRegularCases[index],
        kRegularSteps[index], configured, capture);
    if (status != 0) {
      session.reset();
      ge::GEFinalize();
      return status;
    }
  }
  for (int32_t request = 0; request < kBatchSize; ++request) {
    if (first_tokens[request] < 0 || first_tokens[request] >= kVocabSize) {
      session.reset();
      ge::GEFinalize();
      return 14;
    }
  }
  std::array<int32_t, kBatchSize> early_eos = configured;
  early_eos[0] = static_cast<int32_t>(first_tokens[0]);
  early_eos[2] = static_cast<int32_t>(first_tokens[2]);
  {
    std::ofstream stream(output_root + "/early_eos_tokens.txt",
                         std::ios::trunc);
    for (const auto value : early_eos) stream << value << '\n';
  }
  std::array<std::vector<uint8_t>, 9> early_base;
  if (!LoadInputs(input_root + "/" + kEarlyCase, early_base)) return 15;
  const int early_status = RunHostCase(
      session, early_base, output_root + "/" + kEarlyCase, 4, early_eos,
      nullptr);
  session.reset();
  ge::GEFinalize();
  if (early_status != 0) return early_status;
  std::cout << "HOST_B4_SUITE_COMPLETE early_eos=" << early_eos[0] << ','
            << early_eos[1] << ',' << early_eos[2] << ',' << early_eos[3]
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
  auto data8 = ge::dflow::FlowData("input8", 8);
  auto control = ge::dflow::FlowData("control", 9);
  auto graph_pp = ge::dflow::GraphPp(
      "attempt69e_b4_decoder_graph_pp", [air_path]() {
        ge::Graph graph("Attempt69eB4InvokedDecoder");
        const auto status = graph.LoadFromFile(air_path.c_str());
        std::cout << "DEVICE_AIR_LOAD status=" << status
                  << " valid=" << graph.IsValid() << std::endl;
        return graph;
      });
  graph_pp.SetCompileConfig(graph_config.c_str());
  auto function_pp = ge::dflow::FunctionPp("g4c_b4_resident_epoch_pp")
                         .SetCompileConfig(func_config.c_str());
  function_pp.AddInvokedClosure("decode_graph_0", graph_pp);
  auto node = ge::dflow::FlowNode("g4c_b4_resident_epoch_node", 10, 10);
  node.AddPp(function_pp)
      .SetInput(0, data0)
      .SetInput(1, data1)
      .SetInput(2, data2)
      .SetInput(3, data3)
      .SetInput(4, data4)
      .SetInput(5, data5)
      .SetInput(6, data6)
      .SetInput(7, data7)
      .SetInput(8, data8)
      .SetInput(9, control);
  ge::dflow::FlowGraph flow_graph("attempt69e_b4_resident_epoch");
  std::vector<ge::dflow::FlowOperator> inputs = {
      data0, data1, data2, data3, data4,
      data5, data6, data7, data8, control};
  std::vector<std::pair<ge::dflow::FlowOperator, std::vector<size_t>>> outputs = {
      {node, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}};
  flow_graph.SetInputs(inputs).SetOutputs(outputs);
  return flow_graph;
}

int RunDeviceCase(const std::shared_ptr<ge::Session> &session,
                  const std::array<std::vector<uint8_t>, 9> &base,
                  const std::string &output_dir, int32_t max_steps,
                  const std::array<int32_t, kBatchSize> &eos,
                  Timing *timing = nullptr, bool save_outputs = true) {
  auto buffers = base;
  std::array<int32_t, kControlInputElements> control_values{};
  control_values[0] = max_steps;
  for (int32_t request = 0; request < kBatchSize; ++request) {
    control_values[1 + request] = eos[request];
  }
  control_values[1 + kBatchSize] = 0;
  control_values[2 + kBatchSize] = 0;
  std::vector<uint8_t> control_bytes(sizeof(control_values));
  std::memcpy(control_bytes.data(), control_values.data(), control_bytes.size());
  std::vector<ge::Tensor> inputs;
  inputs.reserve(10);
  for (size_t index = 0; index < kInputSpecs.size(); ++index) {
    inputs.push_back(MakeTensor(buffers[index], kInputSpecs[index].shape,
                                kInputSpecs[index].dtype));
  }
  inputs.push_back(
      MakeTensor(control_bytes, {kControlInputElements}, ge::DT_INT32));
  ge::DataFlowInfo flow_info;
  const auto wall_start = std::chrono::steady_clock::now();
  const auto cpu_start = std::clock();
  auto ret = session->FeedDataFlowGraph(0, inputs, flow_info, kFeedTimeoutMs);
  const auto feed_end = std::chrono::steady_clock::now();
  if (ret != ge::SUCCESS) return 30;
  std::vector<ge::Tensor> outputs;
  ret = session->FetchDataFlowGraph(0, outputs, flow_info, kFetchTimeoutMs);
  const auto wall_end = std::chrono::steady_clock::now();
  const auto cpu_end = std::clock();
  if (ret != ge::SUCCESS || outputs.size() != 10) {
    std::cerr << "DEVICE_FETCH_FAILED ret=" << ret
              << " outputs=" << outputs.size() << std::endl;
    return 31;
  }
  const std::array<const char *, 10> names = {{
      "logits_history.bin", "token_history.bin", "key_cache.bin",
      "value_cache.bin", "final_token.bin", "final_position.bin",
      "final_sequence_length.bin", "final_slot_mapping.bin",
      "final_active_mask.bin", "control.bin"}};
  const std::array<size_t, 10> sizes = {{
      kLogitsHistoryBytes,
      kMaxEpochSteps * kBatchSize * sizeof(int64_t),
      kCacheBytes,
      kCacheBytes,
      kBatchSize * sizeof(int64_t),
      kBatchSize * sizeof(int64_t),
      kBatchSize * sizeof(int32_t),
      kBatchSize * sizeof(int32_t),
      kBatchSize * sizeof(int32_t),
      kControlOutputElements * sizeof(int32_t),
  }};
  const std::array<ge::DataType, 10> dtypes = {{
      ge::DT_FLOAT, ge::DT_INT64, ge::DT_BF16, ge::DT_BF16, ge::DT_INT64,
      ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
  }};
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (outputs[index].GetData() == nullptr ||
        outputs[index].GetSize() != sizes[index] ||
        outputs[index].GetTensorDesc().GetDataType() != dtypes[index]) {
      return 32;
    }
    if (save_outputs &&
        !WriteTensor(output_dir + "/" + names[index], outputs[index],
                     sizes[index], dtypes[index])) {
      return 32;
    }
  }
  const int64_t wall_us =
      std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start)
          .count();
  const int64_t cpu_us = static_cast<int64_t>(
      1000000.0 * static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC);
  if (timing != nullptr) {
    timing->wall_us = wall_us;
    timing->cpu_us = cpu_us;
  }
  if (save_outputs &&
      !WriteRuntime(output_dir + "/runtime.tsv", 1, 1, 1, wall_us, cpu_us)) {
    return 33;
  }
  const int64_t feed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(feed_end - wall_start)
          .count();
  std::cout << "DEVICE_B4_CASE name=" << output_dir
            << " max_steps=" << max_steps
            << " feed_us=" << feed_us << " total_us=" << wall_us
            << " feed_calls=1 fetch_calls=1" << std::endl;
  return 0;
}

int RunDeviceSuite(const std::string &air_path,
                   const std::string &graph_config,
                   const std::string &func_config,
                   const std::string &input_root,
                   const std::string &output_root,
                   const std::array<int32_t, kBatchSize> &early_eos) {
  const char *external_weight_dir = std::getenv("G4_EXTERNAL_WEIGHT_DIR");
  if (external_weight_dir == nullptr ||
      std::strncmp(external_weight_dir, "/dev/shm/", 9) != 0) {
    std::cerr << "G4_EXTERNAL_WEIGHT_DIR must name a /dev/shm child"
              << std::endl;
    return 39;
  }
  auto flow_graph = BuildDeviceFlow(air_path, graph_config, func_config);
  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
      {"ge.externalWeightDir", external_weight_dir},
      {"ge.graphRunMode", "0"}};
  auto ret = ge::GEInitialize(options);
  if (ret != ge::SUCCESS) return 40;
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, flow_graph.ToGeGraph());
  if (ret != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return 41;
  }
  std::array<int32_t, kBatchSize> configured{};
  configured.fill(kConfiguredEos);
  for (size_t index = 0; index < kRegularCases.size(); ++index) {
    std::array<std::vector<uint8_t>, 9> base;
    if (!LoadInputs(input_root + "/" + kRegularCases[index], base)) return 42;
    const int status = RunDeviceCase(
        session, base, output_root + "/" + kRegularCases[index],
        kRegularSteps[index], configured);
    if (status != 0) {
      session.reset();
      ge::GEFinalize();
      return status;
    }
  }
  std::array<std::vector<uint8_t>, 9> early_base;
  if (!LoadInputs(input_root + "/" + kEarlyCase, early_base)) return 43;
  const int early_status = RunDeviceCase(
      session, early_base, output_root + "/" + kEarlyCase, 4, early_eos);
  session.reset();
  ge::GEFinalize();
  if (early_status != 0) return early_status;
  std::cout << "DEVICE_B4_SUITE_COMPLETE early_eos=" << early_eos[0] << ','
            << early_eos[1] << ',' << early_eos[2] << ',' << early_eos[3]
            << std::endl;
  return 0;
}

int RunPerfBlock(const std::string &air_path,
                 const std::string &graph_config,
                 const std::string &func_config,
                 const std::string &input_root,
                 const std::string &output_root, bool host,
                 int32_t block, int32_t repeats) {
  constexpr int32_t kWarmups = 3;
  const std::array<size_t, 3> case_indices = {{1, 2, 3}};
  const std::array<int32_t, 3> steps = {{2, 4, 8}};
  if ((block != 1 && block != 2) ||
      (block == 1 && repeats != 8) ||
      (block == 2 && repeats != 7)) {
    return 60;
  }

  std::map<ge::AscendString, ge::AscendString> options;
  ge::Graph graph;
  if (host) {
    graph = ge::Graph("Attempt70bR1B4HostPerf");
    const auto load_status = graph.LoadFromFile(air_path.c_str());
    if (load_status != ge::GRAPH_SUCCESS || !graph.IsValid()) return 61;
    options = {{"ge.exec.deviceId", "0"},
               {"ge.graphRunMode", "0"},
               {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
  } else {
    const char *external_weight_dir = std::getenv("G4_EXTERNAL_WEIGHT_DIR");
    if (external_weight_dir == nullptr ||
        std::strncmp(external_weight_dir, "/dev/shm/", 9) != 0) {
      return 62;
    }
    graph = BuildDeviceFlow(air_path, graph_config, func_config).ToGeGraph();
    options = {{"ge.exec.deviceId", "0"},
               {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
               {"ge.exec.logicalDeviceId", "[0:0]"},
               {"ge.exec.precision_mode", "must_keep_origin_dtype"},
               {"ge.externalWeightDir", external_weight_dir},
               {"ge.graphRunMode", "0"}};
  }
  auto ret = ge::GEInitialize(options);
  if (ret != ge::SUCCESS) return 63;
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, graph);
  if (ret != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return 64;
  }

  std::array<std::array<std::vector<uint8_t>, 9>, 3> bases;
  for (size_t index = 0; index < case_indices.size(); ++index) {
    if (!LoadInputs(input_root + "/" + kRegularCases[case_indices[index]],
                    bases[index])) {
      session.reset();
      ge::GEFinalize();
      return 65;
    }
  }
  std::array<int32_t, kBatchSize> eos{};
  eos.fill(kConfiguredEos);
  std::ofstream stream(output_root + "/perf-block.tsv", std::ios::trunc);
  if (!stream) {
    session.reset();
    ge::GEFinalize();
    return 66;
  }
  stream << "block\tphase\tk\titeration\torder\tposition\troute\twall_us\tcpu_us"
            "\thost_model_submissions\tfeed_calls\tfetch_calls\n";

  const char *order = block == 1 ? "HD" : "DH";
  const int32_t position = host ? (block == 1 ? 0 : 1)
                                : (block == 1 ? 1 : 0);
  const int32_t measured_offset = block == 1 ? 0 : 8;
  auto run_route = [&](const char *phase, size_t case_index,
                       int32_t iteration) -> int {
    Timing timing;
    int status = host
                     ? RunHostCase(session, bases[case_index], "",
                                   steps[case_index], eos, nullptr, &timing,
                                   false)
                     : RunDeviceCase(session, bases[case_index], "",
                                     steps[case_index], eos, &timing, false);
    if (status != 0) return status;
    stream << block << '\t' << phase << '\t' << steps[case_index] << '\t'
           << iteration << '\t' << order << '\t' << position << '\t'
           << (host ? "host" : "device") << '\t' << timing.wall_us << '\t'
           << timing.cpu_us << '\t' << (host ? steps[case_index] : 1)
           << '\t' << (host ? 0 : 1) << '\t' << (host ? 0 : 1) << '\n';
    stream.flush();
    return stream ? 0 : 67;
  };

  int status = 0;
  for (size_t case_index = 0; case_index < steps.size() && status == 0;
       ++case_index) {
    for (int32_t iteration = 0; iteration < kWarmups && status == 0;
         ++iteration) {
      status = run_route("warmup", case_index, iteration);
    }
    for (int32_t iteration = 0; iteration < repeats && status == 0;
         ++iteration) {
      status = run_route("measure", case_index,
                         measured_offset + iteration);
    }
  }

  stream.close();
  session.reset();
  ge::GEFinalize();
  if (status != 0) return status;
  std::cout << "B4_PERF_BLOCK_COMPLETE route="
            << (host ? "host" : "device") << " block=" << block
            << " warmups=" << kWarmups << " repeats=" << repeats
            << std::endl;
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 11) {
    std::cerr << "usage: g4c_b4_epoch_runner ROUTE AIR GRAPH_CONFIG "
                 "FUNC_CONFIG INPUT_ROOT OUTPUT_ROOT EARLY_EOS_0 EARLY_EOS_1 "
                 "EARLY_EOS_2 EARLY_EOS_3\n";
    return 2;
  }
  const std::string route = argv[1];
  if (route == "host") return RunHostSuite(argv[2], argv[5], argv[6]);
  if (route == "perf-host-block" || route == "perf-device-block") {
    const int32_t block = static_cast<int32_t>(std::stol(argv[7]));
    const int32_t repeats = static_cast<int32_t>(std::stol(argv[8]));
    return RunPerfBlock(argv[2], argv[3], argv[4], argv[5], argv[6],
                        route == "perf-host-block", block, repeats);
  }
  if (route == "device") {
    std::array<int32_t, kBatchSize> early_eos{};
    for (int32_t request = 0; request < kBatchSize; ++request) {
      early_eos[request] =
          static_cast<int32_t>(std::stol(argv[7 + request]));
      if (early_eos[request] < 0 || early_eos[request] >= kVocabSize) {
        return 3;
      }
    }
    return RunDeviceSuite(argv[2], argv[3], argv[4], argv[5], argv[6],
                          early_eos);
  }
  return 2;
}
