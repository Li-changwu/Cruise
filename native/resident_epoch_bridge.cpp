#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <time.h>
#include <vector>

#include "all_ops.h"
#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "ge/ge_data_flow_api.h"
#include "graph/graph.h"
#include "resident_epoch_bridge.h"
#include "resident_epoch_protocol.h"

namespace {
constexpr int32_t kBatchSize = 4;
constexpr int32_t kMaxEpochSteps = 8;
constexpr int32_t kLogicalCapacity = 8;
constexpr int32_t kBlocksPerRequest = 2;
constexpr int32_t kBlockSize = 128;
constexpr int32_t kVocabSize = 152064;
constexpr int32_t kConfiguredEos = 151645;
constexpr int32_t kControlInputElements = 1 + kBatchSize + 2 + kBatchSize;
constexpr int32_t kControlOutputElements = 6 + 4 * kBatchSize + 2 + kBatchSize;
constexpr int32_t kControlExecutedOffset = 6 + 2 * kBatchSize;
constexpr int32_t kControlGenerationOffset = 6 + 4 * kBatchSize + 2;
constexpr size_t kTokenHistoryBytes =
    kMaxEpochSteps * kBatchSize * sizeof(int64_t);
constexpr int64_t kDeclaredInputBytes = 260;
constexpr int64_t kDeclaredOutputBytes = 368;
constexpr int64_t kImportDeclaredInputBytes =
    CRUISE_RESIDENT_IMPORT_INPUT_BYTES;
constexpr int64_t kDeviceIpcDeclaredInputBytes =
    CRUISE_SIDECAR_REQUEST_BYTES + CRUISE_RESIDENT_IPC_METADATA_BYTES;
constexpr int32_t kFeedTimeoutMs = 600000;
constexpr int32_t kFetchTimeoutMs = 3600000;

struct ResidentEpochEngine {
  std::shared_ptr<ge::Session> session;
  std::array<uint8_t, 72> tiling;
  std::mutex execute_mutex;
};

#pragma pack(push, 1)
struct TransferHeader {
  uint64_t magic;
  uint32_t version;
  uint32_t header_bytes;
  uint64_t transfer_id;
  uint64_t payload_bytes;
  uint32_t import_mask;
  int32_t row_generations[kBatchSize];
  uint32_t layers;
  uint32_t batch_size;
  uint32_t block_size;
  uint32_t kv_heads;
  uint32_t head_size;
  uint32_t element_bytes;
  uint32_t checksum;
};
#pragma pack(pop)

static_assert(sizeof(TransferHeader) == CRUISE_RESIDENT_TRANSFER_HEADER_BYTES,
              "resident KV transfer header ABI changed");

std::mutex g_lifecycle_mutex;
bool g_engine_active = false;
ResidentDeviceTransferPrepare g_device_transfer_prepare = nullptr;
ResidentDeviceTransferDestroy g_device_transfer_destroy = nullptr;

bool ReadTiling(const char *path, std::array<uint8_t, 72> &tiling) {
  if (path == nullptr) return false;
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || static_cast<size_t>(stream.tellg()) != tiling.size()) {
    return false;
  }
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(tiling.data()), tiling.size());
  return static_cast<bool>(stream);
}

bool ReadTransfer(const char *path, uint64_t transfer_id,
                  const int32_t *input_row_generations,
                  std::vector<uint8_t> &payload, int32_t &import_mask,
                  uint32_t &expected_checksum) {
  if (path == nullptr || input_row_generations == nullptr || transfer_id == 0 ||
      std::strncmp(path, "/dev/shm/", 9) != 0) {
    return false;
  }
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  const size_t expected_bytes = CRUISE_RESIDENT_TRANSFER_HEADER_BYTES +
                                CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES;
  if (!stream || static_cast<size_t>(stream.tellg()) != expected_bytes) {
    return false;
  }
  stream.seekg(0, std::ios::beg);
  TransferHeader header{};
  stream.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!stream || header.magic != CRUISE_RESIDENT_TRANSFER_MAGIC ||
      header.version != CRUISE_RESIDENT_TRANSFER_VERSION ||
      header.header_bytes != CRUISE_RESIDENT_TRANSFER_HEADER_BYTES ||
      header.transfer_id != transfer_id ||
      header.payload_bytes != CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES ||
      header.import_mask == 0 || header.import_mask >= (1U << kBatchSize) ||
      header.layers != 28 || header.batch_size != kBatchSize ||
      header.block_size != kBlockSize || header.kv_heads != 4 ||
      header.head_size != 128 || header.element_bytes != sizeof(uint16_t) ||
      header.checksum == 0) {
    return false;
  }
  for (int32_t row = 0; row < kBatchSize; ++row) {
    const bool selected = (header.import_mask & (1U << row)) != 0;
    if (selected) {
      if (header.row_generations[row] <= 0 ||
          header.row_generations[row] != input_row_generations[row]) {
        return false;
      }
    } else if (header.row_generations[row] != 0) {
      return false;
    }
  }
  payload.resize(CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES);
  stream.read(reinterpret_cast<char *>(payload.data()), payload.size());
  if (!stream) return false;
  import_mask = static_cast<int32_t>(header.import_mask);
  expected_checksum = header.checksum;
  return true;
}

bool ValidateIpcMetadata(const ResidentEpochIpcMetadata *metadata,
                         const int32_t *input_row_generations) {
  if (metadata == nullptr || input_row_generations == nullptr ||
      metadata->magic != CRUISE_RESIDENT_IPC_METADATA_MAGIC ||
      metadata->version != CRUISE_RESIDENT_IPC_METADATA_VERSION ||
      metadata->import_mask == 0 ||
      metadata->import_mask >= (1U << kBatchSize) ||
      metadata->segment_count == 0 ||
      metadata->segment_count > CRUISE_RESIDENT_IPC_MAX_SEGMENTS ||
      metadata->reserved != 0) {
    return false;
  }
  uint64_t selected_rows = 0;
  for (int32_t row = 0; row < kBatchSize; ++row) {
    const bool selected = (metadata->import_mask & (1U << row)) != 0;
    if (selected) {
      ++selected_rows;
      if (metadata->row_generations[row] <= 0 ||
          metadata->row_generations[row] != input_row_generations[row] ||
          metadata->block_ids[row] < 0) {
        return false;
      }
    } else if (metadata->row_generations[row] != 0 ||
               metadata->block_ids[row] != 0) {
      return false;
    }
  }
  const uint64_t payload_bytes = CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES;
  const uint64_t cache_bytes = payload_bytes / 2;
  const uint64_t block_bytes = CRUISE_RESIDENT_KV_BLOCK_BYTES;
  uint64_t total_bytes = 0;
  struct Interval {
    uint64_t start;
    uint64_t end;
  };
  std::array<Interval, CRUISE_RESIDENT_IPC_MAX_SEGMENTS> coverage{};
  for (uint32_t index = 0; index < metadata->segment_count; ++index) {
    const auto &segment = metadata->segments[index];
    if (segment.source_allocation_bytes == 0 || segment.copy_bytes == 0 ||
        segment.source_offset > segment.source_allocation_bytes ||
        segment.copy_bytes >
            segment.source_allocation_bytes - segment.source_offset ||
        segment.destination_offset > payload_bytes ||
        segment.copy_bytes > payload_bytes - segment.destination_offset) {
      return false;
    }
    const uint64_t cache_offset = segment.destination_offset % cache_bytes;
    const uint64_t layer_row = cache_offset / block_bytes;
    const uint64_t row = layer_row % kBatchSize;
    const uint64_t block_offset = cache_offset % block_bytes;
    if ((metadata->import_mask & (1U << row)) == 0 ||
        segment.copy_bytes > block_bytes - block_offset) {
      return false;
    }
    if (total_bytes > payload_bytes - segment.copy_bytes) return false;
    total_bytes += segment.copy_bytes;
    coverage[index] = {segment.destination_offset,
                       segment.destination_offset + segment.copy_bytes};
    const size_t key_length =
        strnlen(segment.key, CRUISE_RESIDENT_IPC_KEY_BYTES);
    if (key_length == 0) return false;
    if (key_length < CRUISE_RESIDENT_IPC_KEY_BYTES &&
        std::any_of(segment.key + key_length + 1,
                    segment.key + CRUISE_RESIDENT_IPC_KEY_BYTES,
                    [](char value) { return value != '\0'; })) return false;
  }
  for (uint32_t index = metadata->segment_count;
       index < CRUISE_RESIDENT_IPC_MAX_SEGMENTS; ++index) {
    const auto &segment = metadata->segments[index];
    if (segment.source_offset != 0 || segment.source_allocation_bytes != 0 ||
        segment.destination_offset != 0 || segment.copy_bytes != 0 ||
        std::any_of(segment.key,
                    segment.key + CRUISE_RESIDENT_IPC_KEY_BYTES,
                    [](char value) { return value != '\0'; })) {
      return false;
    }
  }
  std::sort(coverage.begin(), coverage.begin() + metadata->segment_count,
            [](const Interval &left, const Interval &right) {
              return left.start < right.start;
            });
  uint32_t interval_index = 0;
  for (uint32_t cache_index = 0; cache_index < 2; ++cache_index) {
    const uint64_t cache_base = cache_index * cache_bytes;
    for (uint32_t layer = 0; layer < 28; ++layer) {
      for (uint32_t row = 0; row < kBatchSize; ++row) {
        if ((metadata->import_mask & (1U << row)) == 0) continue;
        const uint64_t block_start =
            cache_base + (layer * kBatchSize + row) * block_bytes;
        const uint64_t block_end = block_start + block_bytes;
        uint64_t cursor = block_start;
        uint32_t block_segments = 0;
        while (interval_index < metadata->segment_count &&
               coverage[interval_index].start < block_end) {
          const auto &span = coverage[interval_index];
          if (span.start != cursor || span.end > block_end ||
              ++block_segments > CRUISE_RESIDENT_IPC_MAX_SEGMENTS_PER_BLOCK) {
            return false;
          }
          cursor = span.end;
          ++interval_index;
        }
        if (cursor != block_end) return false;
      }
    }
  }
  return interval_index == metadata->segment_count &&
         total_bytes == selected_rows * CRUISE_RESIDENT_IPC_KEY_COUNT *
                            CRUISE_RESIDENT_KV_BLOCK_BYTES;
}

ge::Tensor MakeTensor(std::vector<uint8_t> &data,
                      const std::vector<int64_t> &shape,
                      ge::DataType dtype) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype));
  tensor.SetData(data.data(), data.size());
  return tensor;
}

ge::FlowMsgPtr MakeHostFlowMsg(std::vector<uint8_t> &data,
                               const std::vector<int64_t> &shape,
                               ge::DataType dtype) {
  return ge::FlowBufferFactory::ToFlowMsg(MakeTensor(data, shape, dtype));
}

bool AppendHostFlowMsg(std::vector<ge::FlowMsgPtr> &inputs,
                       std::vector<uint8_t> &data,
                       const std::vector<int64_t> &shape,
                       ge::DataType dtype) {
  auto message = MakeHostFlowMsg(data, shape, dtype);
  if (message == nullptr || message->GetTensor() == nullptr) return false;
  inputs.push_back(message);
  return true;
}

bool IsOutput(const ge::FlowMsgPtr &message, size_t bytes,
              ge::DataType dtype) {
  const auto *tensor = message == nullptr ? nullptr : message->GetTensor();
  return tensor != nullptr && tensor->GetData() != nullptr &&
         tensor->GetSize() == bytes &&
         tensor->GetDataType() == dtype;
}

int64_t ProcessCpuUs() {
  timespec value{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) return -1;
  return static_cast<int64_t>(value.tv_sec) * 1000000LL +
         static_cast<int64_t>(value.tv_nsec) / 1000LL;
}

bool ReplaceLogitsWithDeviceGreedyTokens(ge::Graph &graph) {
  ge::GNodePtr net_output;
  for (const auto &node : graph.GetAllNodes()) {
    ge::AscendString type;
    if (node.GetType(type) != ge::GRAPH_SUCCESS) return false;
    const char *type_name = type.GetString();
    if (type_name == nullptr || std::strcmp(type_name, "NetOutput") != 0) {
      continue;
    }
    if (net_output != nullptr || node.GetInputsSize() != 4) return false;
    net_output = std::make_shared<ge::GNode>(node);
  }
  if (net_output == nullptr) return false;

  ge::GNodePtr logits;
  int32_t logits_port = -1;
  std::vector<std::pair<ge::GNode, int32_t>> outputs;
  for (size_t index = 0; index < net_output->GetInputsSize(); ++index) {
    const auto source = net_output->GetInDataNodesAndPortIndexs(index);
    if (source.first == nullptr || source.second < 0) return false;
    if (index == 0) {
      logits = source.first;
      logits_port = source.second;
    } else {
      outputs.emplace_back(*source.first, source.second);
    }
  }
  if (logits == nullptr || logits_port < 0 || outputs.size() != 3) return false;

  ge::op::ArgMaxWithValue token_ids("resident_epoch_device_greedy_tokens");
  token_ids.set_attr_dimension(static_cast<int64_t>(-1));
  token_ids.set_attr_indice_dtype(ge::DT_INT64);
  ge::GNode token_node = graph.AddNodeByOp(token_ids);
  if (token_node.GetInputsSize() != 1 || token_node.GetOutputsSize() != 2 ||
      graph.AddDataEdge(*logits, logits_port, token_node, 0) !=
          ge::GRAPH_SUCCESS) {
    return false;
  }
  outputs.insert(outputs.begin(), {token_node, 0});
  return graph.SetOutputs(outputs) == ge::GRAPH_SUCCESS && graph.IsValid();
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
  auto control = ge::dflow::FlowData("control", 7);
  auto graph_pp = ge::dflow::GraphPp(
      "attempt69e_b4_decoder_graph_pp", [air_path]() {
        ge::Graph graph("Attempt69eB4InvokedDecoder");
        const auto status = graph.LoadFromFile(air_path.c_str());
        std::cout << "ATTEMPT71_AIR_LOAD status=" << status
                  << " valid=" << graph.IsValid() << std::endl;
        if (status != ge::GRAPH_SUCCESS || !graph.IsValid() ||
            !ReplaceLogitsWithDeviceGreedyTokens(graph)) {
          std::cerr << "resident epoch failed to install Device greedy "
                       "sampling output"
                    << std::endl;
          return ge::Graph("ResidentEpochInvalidDecoder");
        }
        return graph;
      });
  graph_pp.SetCompileConfig(graph_config.c_str());
  auto function_pp = ge::dflow::FunctionPp("g4c_b4_resident_epoch_pp")
                         .SetCompileConfig(func_config.c_str());
  function_pp.AddInvokedClosure("decode_graph_0", graph_pp);
  auto node = ge::dflow::FlowNode("g4c_b4_resident_epoch_node", 8, 2);
  node.AddPp(function_pp)
      .SetInput(0, data0)
      .SetInput(1, data1)
      .SetInput(2, data2)
      .SetInput(3, data3)
      .SetInput(4, data4)
      .SetInput(5, data5)
      .SetInput(6, data6)
      .SetInput(7, control);
  ge::dflow::FlowGraph flow_graph("attempt69e_b4_resident_epoch");
  std::vector<ge::dflow::FlowOperator> inputs = {
      data0, data1, data2, data3, data4, data5, data6, control};
  std::vector<std::pair<ge::dflow::FlowOperator, std::vector<size_t>>> outputs = {
      {node, {0, 1}}};
  flow_graph.SetInputs(inputs).SetOutputs(outputs);
  return flow_graph;
}

int32_t ComputeSlot(int32_t row, int64_t position) {
  if (row < 0 || row >= kBatchSize || position < 0 ||
      position >= kLogicalCapacity) {
    return -1;
  }
  const int32_t physical_block = row * kBlocksPerRequest;
  return physical_block * kBlockSize + static_cast<int32_t>(position);
}
}  // namespace

extern "C" int32_t resident_epoch_install_device_transfer(
    ResidentDeviceTransferPrepare prepare,
    ResidentDeviceTransferDestroy destroy) {
  if (prepare == nullptr || destroy == nullptr) return 1;
  std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mutex);
  if (!g_engine_active || g_device_transfer_prepare != nullptr ||
      g_device_transfer_destroy != nullptr) {
    return 2;
  }
  g_device_transfer_prepare = prepare;
  g_device_transfer_destroy = destroy;
  return 0;
}

extern "C" void *resident_epoch_create(
    const char *air_path, const char *graph_config, const char *func_config,
    const char *external_weight_dir, const char *tiling_path,
    int32_t *status) {
  if (status == nullptr) return nullptr;
  *status = 1;
  std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mutex);
  if (g_engine_active || air_path == nullptr || graph_config == nullptr ||
      func_config == nullptr || external_weight_dir == nullptr ||
      std::strncmp(external_weight_dir, "/dev/shm/", 9) != 0) {
    return nullptr;
  }
  std::unique_ptr<ResidentEpochEngine> engine(new ResidentEpochEngine());
  if (!ReadTiling(tiling_path, engine->tiling)) {
    *status = 2;
    return nullptr;
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
  if (ret != ge::SUCCESS) {
    *status = 3;
    return nullptr;
  }
  engine->session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  auto graph = flow_graph.ToGeGraph();
  std::cout << "ATTEMPT71_FLOW_GRAPH valid=" << graph.IsValid() << std::endl;
  if (!graph.IsValid()) {
    engine->session.reset();
    ge::GEFinalize();
    *status = 5;
    return nullptr;
  }
  ret = engine->session->AddGraph(0, graph);
  if (ret != ge::SUCCESS) {
    engine->session.reset();
    ge::GEFinalize();
    *status = 4;
    return nullptr;
  }
  g_engine_active = true;
  *status = 0;
  return engine.release();
}

extern "C" int32_t resident_epoch_execute(
    void *opaque, int32_t request_count, int32_t max_steps,
    const int64_t *input_token_ids, const int64_t *input_positions,
    const int32_t *input_sequence_lengths, const int32_t *input_eos_token_ids,
    const int32_t *input_row_generations,
    int64_t *output_token_ids, int32_t *output_executed,
    int32_t *output_row_generations,
    int32_t *output_model_calls, int32_t *output_device_status,
    int32_t *output_feed_calls, int32_t *output_fetch_calls,
    int32_t *output_commit_state, int32_t *output_kv_import_checksum,
    int64_t *output_wall_us, int64_t *output_native_cpu_us,
    int64_t *output_device_kv_transfer_wall_us,
    int64_t *output_device_kv_transfer_cpu_us,
    int64_t *output_declared_input_bytes,
    int64_t *output_declared_output_bytes,
    const char *transfer_path, uint64_t transfer_id,
    const ResidentEpochIpcMetadata *ipc_metadata) {
  if (output_commit_state == nullptr) return 10;
  *output_commit_state = CRUISE_EPOCH_PREPARED;
  const bool direct_device_import = ipc_metadata != nullptr;
  const bool importing = direct_device_import;
  if (opaque == nullptr || request_count < 1 || request_count > kBatchSize ||
      max_steps < 1 || max_steps > kMaxEpochSteps ||
      input_token_ids == nullptr || input_positions == nullptr ||
      input_sequence_lengths == nullptr || input_eos_token_ids == nullptr ||
      input_row_generations == nullptr ||
      output_token_ids == nullptr || output_executed == nullptr ||
      output_row_generations == nullptr ||
      output_model_calls == nullptr || output_device_status == nullptr ||
      output_feed_calls == nullptr || output_fetch_calls == nullptr ||
      output_kv_import_checksum == nullptr ||
      output_wall_us == nullptr || output_native_cpu_us == nullptr ||
      output_device_kv_transfer_wall_us == nullptr ||
      output_device_kv_transfer_cpu_us == nullptr ||
      output_declared_input_bytes == nullptr ||
      output_declared_output_bytes == nullptr ||
      transfer_path != nullptr || (importing && transfer_id == 0) ||
      (!importing && transfer_id != 0)) {
    return 10;
  }
  auto *engine = static_cast<ResidentEpochEngine *>(opaque);
  std::lock_guard<std::mutex> execute_lock(engine->execute_mutex);
  int32_t import_mask = 0;
  if (direct_device_import) {
    if (!ValidateIpcMetadata(ipc_metadata, input_row_generations) ||
        g_device_transfer_prepare == nullptr) {
      return 35;
    }
    import_mask = static_cast<int32_t>(ipc_metadata->import_mask);
  }
  *output_model_calls = 0;
  *output_device_status = -1;
  *output_feed_calls = 0;
  *output_fetch_calls = 0;
  *output_kv_import_checksum = 0;
  *output_wall_us = 0;
  *output_native_cpu_us = 0;
  *output_device_kv_transfer_wall_us = 0;
  *output_device_kv_transfer_cpu_us = 0;
  *output_declared_input_bytes =
      direct_device_import ? kDeviceIpcDeclaredInputBytes : kDeclaredInputBytes;
  *output_declared_output_bytes = kDeclaredOutputBytes;
  const int64_t cpu_start = ProcessCpuUs();
  std::fill(output_token_ids,
            output_token_ids + kBatchSize * kMaxEpochSteps, -1);
  std::fill(output_executed, output_executed + kBatchSize, 0);
  std::fill(output_row_generations,
            output_row_generations + kBatchSize, 0);

  std::vector<uint8_t> token_buffer(kBatchSize * sizeof(int64_t), 0);
  std::vector<uint8_t> position_buffer(kBatchSize * sizeof(int64_t), 0);
  std::vector<uint8_t> length_buffer(kBatchSize * sizeof(int32_t), 0);
  std::vector<uint8_t> slot_buffer(kBatchSize * sizeof(int32_t), 0);
  std::vector<uint8_t> active_buffer(kBatchSize * sizeof(int32_t), 0);
  std::vector<uint8_t> block_buffer(
      kBatchSize * kBlocksPerRequest * sizeof(int32_t), 0);
  std::vector<uint8_t> tiling_buffer(engine->tiling.size(), 0);
  auto *tokens = reinterpret_cast<int64_t *>(token_buffer.data());
  auto *positions = reinterpret_cast<int64_t *>(position_buffer.data());
  auto *lengths = reinterpret_cast<int32_t *>(length_buffer.data());
  auto *slots = importing
                    ? nullptr
                    : reinterpret_cast<int32_t *>(slot_buffer.data());
  auto *active = reinterpret_cast<int32_t *>(active_buffer.data());
  auto *blocks = reinterpret_cast<int32_t *>(block_buffer.data());
  for (int32_t row = 0; row < kBatchSize; ++row) {
    blocks[row * kBlocksPerRequest] = row * kBlocksPerRequest;
    blocks[row * kBlocksPerRequest + 1] = row * kBlocksPerRequest + 1;
    tokens[row] = 0;
    positions[row] = 0;
    lengths[row] = 0;
    if (slots != nullptr) slots[row] = ComputeSlot(row, 0);
    active[row] = 0;
  }
  int32_t active_count = 0;
  for (int32_t row = 0; row < kBatchSize; ++row) {
    if (input_row_generations[row] == 0) continue;
    ++active_count;
    if (input_row_generations[row] < 0 ||
        input_token_ids[row] < 0 || input_token_ids[row] >= kVocabSize ||
        input_positions[row] < 0 ||
        input_positions[row] + max_steps > kLogicalCapacity ||
        input_sequence_lengths[row] != input_positions[row] + 1 ||
        input_eos_token_ids[row] < 0 ||
        input_eos_token_ids[row] >= kVocabSize) {
      return 11;
    }
    tokens[row] = input_token_ids[row];
    positions[row] = input_positions[row];
    lengths[row] = input_sequence_lengths[row];
    if (slots != nullptr) slots[row] = ComputeSlot(row, input_positions[row]);
    active[row] = 1;
  }
  if (active_count != request_count) return 12;
  std::memcpy(tiling_buffer.data(), engine->tiling.data(),
              engine->tiling.size());

  std::array<int32_t, kControlInputElements> control{};
  control[0] = max_steps;
  for (int32_t row = 0; row < kBatchSize; ++row) {
    control[1 + row] = input_row_generations[row] != 0
                           ? input_eos_token_ids[row]
                           : kConfiguredEos;
  }
  control[1 + kBatchSize] = 0;
  control[2 + kBatchSize] =
      importing ? CRUISE_RESIDENT_IMPORT_GRAPH_FLAG | import_mask : 0;
  for (int32_t row = 0; row < kBatchSize; ++row) {
    control[3 + kBatchSize + row] = input_row_generations[row];
  }
  std::vector<uint8_t> control_bytes(sizeof(control));
  std::memcpy(control_bytes.data(), control.data(), control_bytes.size());

  std::vector<ge::FlowMsgPtr> inputs;
  inputs.reserve(8);
  if (importing) {
    if (direct_device_import) {
      auto device_input = ge::FlowBufferFactory::AllocTensorMsg(
          {CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES}, ge::DT_UINT8);
      if (device_input == nullptr || device_input->GetTensor() == nullptr ||
          device_input->GetTensor()->GetData() == nullptr ||
          device_input->GetTensor()->GetSize() !=
              CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES) {
        return 35;
      }
      const auto transfer_wall_start = std::chrono::steady_clock::now();
      const int64_t transfer_cpu_start = ProcessCpuUs();
      const int32_t transfer_status = g_device_transfer_prepare(
          ipc_metadata, device_input->GetTensor()->GetData(),
          device_input->GetTensor()->GetSize());
      const auto transfer_wall_end = std::chrono::steady_clock::now();
      const int64_t transfer_cpu_end = ProcessCpuUs();
      *output_device_kv_transfer_wall_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              transfer_wall_end - transfer_wall_start)
              .count();
      if (transfer_cpu_start >= 0 && transfer_cpu_end >= transfer_cpu_start) {
        *output_device_kv_transfer_cpu_us = transfer_cpu_end - transfer_cpu_start;
      }
      if (transfer_status != 0) return 35;
      inputs.push_back(device_input);
    }
  }
  if (!AppendHostFlowMsg(inputs, token_buffer, {4, 1}, ge::DT_INT64) ||
      !AppendHostFlowMsg(inputs, position_buffer, {4}, ge::DT_INT64) ||
      !AppendHostFlowMsg(inputs, length_buffer, {4, 1}, ge::DT_INT32)) {
    return 36;
  }
  if (!importing) {
    if (!AppendHostFlowMsg(inputs, slot_buffer, {4}, ge::DT_INT32)) {
      return 36;
    }
  }
  if (!AppendHostFlowMsg(inputs, active_buffer, {4}, ge::DT_INT32) ||
      !AppendHostFlowMsg(inputs, block_buffer, {4, 2}, ge::DT_INT32) ||
      !AppendHostFlowMsg(inputs, tiling_buffer, {72}, ge::DT_UINT8) ||
      !AppendHostFlowMsg(inputs, control_bytes, {kControlInputElements},
                         ge::DT_INT32)) {
    return 36;
  }
  const auto wall_start = std::chrono::steady_clock::now();
  *output_commit_state = CRUISE_EPOCH_EXECUTING;
  auto ret = engine->session->FeedDataFlowGraph(0, inputs, kFeedTimeoutMs);
  *output_feed_calls = 1;
  if (ret != ge::SUCCESS) return 30;
  std::vector<ge::FlowMsgPtr> outputs;
  ret = engine->session->FetchDataFlowGraph(0, outputs, kFetchTimeoutMs);
  *output_fetch_calls = 1;
  const auto wall_end = std::chrono::steady_clock::now();
  *output_wall_us =
      std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start)
          .count();
  if (ret != ge::SUCCESS || outputs.size() != 2) return 31;
  if (!IsOutput(outputs[0], kTokenHistoryBytes, ge::DT_INT64) ||
      !IsOutput(outputs[1], kControlOutputElements * sizeof(int32_t),
                 ge::DT_INT32)) {
    return 32;
  }
  const auto *history =
      reinterpret_cast<const int64_t *>(outputs[0]->GetTensor()->GetData());
  const auto *result_control =
      reinterpret_cast<const int32_t *>(outputs[1]->GetTensor()->GetData());
  *output_device_status = result_control[3];
  *output_model_calls = result_control[4];
  *output_kv_import_checksum = result_control[5];
  if (*output_device_status == 0) {
    if (direct_device_import && *output_kv_import_checksum == 0) return 34;
  }
  for (int32_t row = 0; row < kBatchSize; ++row) {
    const int32_t executed = result_control[kControlExecutedOffset + row];
    if (executed < 0 || executed > max_steps) return 33;
    output_executed[row] = executed;
    output_row_generations[row] =
        result_control[kControlGenerationOffset + row];
    for (int32_t step = 0; step < executed; ++step) {
      output_token_ids[row * kMaxEpochSteps + step] =
          history[step * kBatchSize + row];
    }
  }
  const int64_t cpu_end = ProcessCpuUs();
  if (cpu_start >= 0 && cpu_end >= cpu_start) {
    *output_native_cpu_us = cpu_end - cpu_start;
  }
  *output_commit_state = CRUISE_EPOCH_COMMITTED;
  return 0;
}

extern "C" void resident_epoch_destroy(void *opaque) {
  if (opaque == nullptr) return;
  std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mutex);
  auto *engine = static_cast<ResidentEpochEngine *>(opaque);
  {
    std::lock_guard<std::mutex> execute_lock(engine->execute_mutex);
    if (g_device_transfer_destroy != nullptr) {
      g_device_transfer_destroy();
      g_device_transfer_prepare = nullptr;
      g_device_transfer_destroy = nullptr;
    }
    engine->session.reset();
    ge::GEFinalize();
  }
  delete engine;
  g_engine_active = false;
}
