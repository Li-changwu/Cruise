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

#include "acl/acl_rt.h"
#include "all_ops.h"
#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "resident_epoch_bridge.h"
#include "resident_epoch_protocol.h"

namespace {
constexpr int32_t kBatchSize = 4;
constexpr int32_t kMaxEpochSteps = 8;
constexpr int32_t kLogicalCapacity = 8;
constexpr int32_t kPhysicalBlocks = 8;
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

struct InputSpec {
  std::vector<int64_t> shape;
  ge::DataType dtype;
  size_t bytes;
};

const std::array<InputSpec, 7> kInputSpecs = {{
    {{4, 1}, ge::DT_INT64, kBatchSize * sizeof(int64_t)},
    {{4}, ge::DT_INT64, kBatchSize * sizeof(int64_t)},
    {{4, 1}, ge::DT_INT32, kBatchSize * sizeof(int32_t)},
    {{4}, ge::DT_INT32, kBatchSize * sizeof(int32_t)},
    {{4}, ge::DT_INT32, kBatchSize * sizeof(int32_t)},
    {{4, 2}, ge::DT_INT32,
     kBatchSize * kBlocksPerRequest * sizeof(int32_t)},
    {{72}, ge::DT_UINT8, 72},
}};

struct ResidentEpochEngine {
  std::shared_ptr<ge::Session> session;
  std::array<uint8_t, 72> tiling;
  std::mutex execute_mutex;
  void *device_import_payload = nullptr;
  std::map<std::string, void *> ipc_imports;
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
      metadata->source_bytes == 0 || metadata->import_mask == 0 ||
      metadata->import_mask >= (1U << kBatchSize)) {
    return false;
  }
  for (int32_t row = 0; row < kBatchSize; ++row) {
    const bool selected = (metadata->import_mask & (1U << row)) != 0;
    if (selected) {
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
  for (int32_t index = 0; index < CRUISE_RESIDENT_IPC_KEY_COUNT; ++index) {
    if (std::memchr(metadata->keys[index], '\0',
                    CRUISE_RESIDENT_IPC_KEY_BYTES) == nullptr) {
      return false;
    }
  }
  return true;
}

void *ImportIpcMemory(ResidentEpochEngine *engine, const char *key) {
  if (engine == nullptr || key == nullptr || *key == '\0') return nullptr;
  const std::string key_string(key);
  const auto existing = engine->ipc_imports.find(key_string);
  if (existing != engine->ipc_imports.end()) return existing->second;
  void *device_ptr = nullptr;
  const auto status = aclrtIpcMemImportByKey(
      &device_ptr, key, ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT);
  if (status != ACL_SUCCESS || device_ptr == nullptr) return nullptr;
  engine->ipc_imports.emplace(key_string, device_ptr);
  return device_ptr;
}

bool PrepareDeviceIpcPayload(ResidentEpochEngine *engine,
                             const ResidentEpochIpcMetadata *metadata,
                             void **payload_out) {
  if (engine == nullptr || metadata == nullptr || payload_out == nullptr) {
    return false;
  }
  if (metadata->source_bytes <
      static_cast<uint64_t>(kPhysicalBlocks) * CRUISE_RESIDENT_KV_BLOCK_BYTES) {
    return false;
  }
  if (engine->device_import_payload == nullptr) {
    if (aclrtMalloc(&engine->device_import_payload,
                    CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES,
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      return false;
    }
  }
  if (aclrtMemset(engine->device_import_payload,
                  CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES, 0,
                  CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES) != ACL_SUCCESS) {
    return false;
  }
  const size_t cache_bytes = CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES / 2;
  const size_t block_bytes = CRUISE_RESIDENT_KV_BLOCK_BYTES;
  auto *destination = static_cast<uint8_t *>(engine->device_import_payload);
  for (int32_t layer = 0; layer < 28; ++layer) {
    const char *key_key = metadata->keys[layer * 2];
    const char *value_key = metadata->keys[layer * 2 + 1];
    void *key_source = ImportIpcMemory(engine, key_key);
    void *value_source = ImportIpcMemory(engine, value_key);
    if (key_source == nullptr || value_source == nullptr) return false;
    for (int32_t row = 0; row < kBatchSize; ++row) {
      if ((metadata->import_mask & (1U << row)) == 0) continue;
      const size_t source_offset =
          static_cast<size_t>(metadata->block_ids[row]) * block_bytes;
      const size_t row_offset =
          (static_cast<size_t>(layer) * kBatchSize + row) * block_bytes;
      if (aclrtMemcpy(destination + row_offset, block_bytes,
                      static_cast<uint8_t *>(key_source) + source_offset,
                      block_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE) != ACL_SUCCESS ||
          aclrtMemcpy(destination + cache_bytes + row_offset, block_bytes,
                      static_cast<uint8_t *>(value_source) + source_offset,
                      block_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE) != ACL_SUCCESS) {
        return false;
      }
    }
  }
  *payload_out = engine->device_import_payload;
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

ge::Tensor MakeDeviceTensor(void *data, size_t bytes,
                            const std::vector<int64_t> &shape,
                            ge::DataType dtype) {
  ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
  desc.SetPlacement(ge::kPlacementDevice);
  ge::Tensor tensor(desc);
  tensor.SetData(static_cast<uint8_t *>(data), bytes,
                 [](uint8_t *) {});
  tensor.SetPlacement(ge::kPlacementDevice);
  return tensor;
}

bool IsOutput(const ge::Tensor &tensor, size_t bytes, ge::DataType dtype) {
  return tensor.GetData() != nullptr && tensor.GetSize() == bytes &&
         tensor.GetTensorDesc().GetDataType() == dtype;
}

int64_t ProcessCpuUs() {
  timespec value{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) return -1;
  return static_cast<int64_t>(value.tv_sec) * 1000000LL +
         static_cast<int64_t>(value.tv_nsec) / 1000LL;
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
  const auto acl_init_status = aclInit(nullptr);
  if (acl_init_status != ACL_SUCCESS &&
      acl_init_status != ACL_ERROR_REPEAT_INITIALIZE) {
    ge::GEFinalize();
    *status = 6;
    return nullptr;
  }
  if (aclrtSetDevice(0) != ACL_SUCCESS) {
    ge::GEFinalize();
    *status = 7;
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
    int64_t *output_declared_input_bytes,
    int64_t *output_declared_output_bytes,
    const char *transfer_path, uint64_t transfer_id,
    const ResidentEpochIpcMetadata *ipc_metadata) {
  if (output_commit_state == nullptr) return 10;
  *output_commit_state = CRUISE_EPOCH_PREPARED;
  const bool direct_device_import = ipc_metadata != nullptr;
  const bool importing = transfer_path != nullptr || direct_device_import;
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
      output_declared_input_bytes == nullptr ||
      output_declared_output_bytes == nullptr ||
      (importing && transfer_id == 0) || (!importing && transfer_id != 0) ||
      (transfer_path != nullptr && direct_device_import)) {
    return 10;
  }
  auto *engine = static_cast<ResidentEpochEngine *>(opaque);
  std::lock_guard<std::mutex> execute_lock(engine->execute_mutex);
  std::vector<uint8_t> transfer_payload;
  int32_t import_mask = 0;
  uint32_t expected_import_checksum = 0;
  if (importing &&
      !direct_device_import &&
      !ReadTransfer(transfer_path, transfer_id, input_row_generations,
                    transfer_payload, import_mask, expected_import_checksum)) {
    return 13;
  }
  void *device_import_payload = nullptr;
  if (direct_device_import) {
    if (!ValidateIpcMetadata(ipc_metadata, input_row_generations) ||
        !PrepareDeviceIpcPayload(engine, ipc_metadata,
                                 &device_import_payload)) {
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
  *output_declared_input_bytes =
      direct_device_import
          ? kDeviceIpcDeclaredInputBytes
          : (importing ? kImportDeclaredInputBytes : kDeclaredInputBytes);
  *output_declared_output_bytes = kDeclaredOutputBytes;
  const int64_t cpu_start = ProcessCpuUs();
  std::fill(output_token_ids,
            output_token_ids + kBatchSize * kMaxEpochSteps, -1);
  std::fill(output_executed, output_executed + kBatchSize, 0);
  std::fill(output_row_generations,
            output_row_generations + kBatchSize, 0);

  std::array<std::vector<uint8_t>, 7> buffers;
  if (importing && !direct_device_import) {
    buffers[0] = std::move(transfer_payload);
    buffers[1].resize(kBatchSize * sizeof(int64_t), 0);
    buffers[2].resize(kBatchSize * sizeof(int64_t), 0);
    buffers[3].resize(kBatchSize * sizeof(int32_t), 0);
    buffers[4].resize(kBatchSize * sizeof(int32_t), 0);
    buffers[5].resize(kBatchSize * kBlocksPerRequest * sizeof(int32_t), 0);
    buffers[6].resize(engine->tiling.size(), 0);
  } else {
    for (size_t index = 0; index < buffers.size(); ++index) {
      buffers[index].resize(kInputSpecs[index].bytes, 0);
    }
  }
  auto *tokens = reinterpret_cast<int64_t *>(
      buffers[importing ? 1 : 0].data());
  auto *positions = reinterpret_cast<int64_t *>(
      buffers[importing ? 2 : 1].data());
  auto *lengths = reinterpret_cast<int32_t *>(
      buffers[importing ? 3 : 2].data());
  auto *slots = importing
                    ? nullptr
                    : reinterpret_cast<int32_t *>(buffers[3].data());
  auto *active = reinterpret_cast<int32_t *>(buffers[4].data());
  auto *blocks = reinterpret_cast<int32_t *>(buffers[5].data());
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
  std::memcpy(buffers[6].data(), engine->tiling.data(), engine->tiling.size());

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

  std::vector<ge::Tensor> inputs;
  inputs.reserve(8);
  if (importing) {
    if (direct_device_import) {
      inputs.push_back(MakeDeviceTensor(
          device_import_payload, CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES,
          {CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES}, ge::DT_UINT8));
    } else {
      inputs.push_back(MakeTensor(
          buffers[0], {CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES}, ge::DT_UINT8));
    }
    inputs.push_back(MakeTensor(buffers[1], {4, 1}, ge::DT_INT64));
    inputs.push_back(MakeTensor(buffers[2], {4}, ge::DT_INT64));
    inputs.push_back(MakeTensor(buffers[3], {4, 1}, ge::DT_INT32));
    inputs.push_back(MakeTensor(buffers[4], {4}, ge::DT_INT32));
    inputs.push_back(MakeTensor(buffers[5], {4, 2}, ge::DT_INT32));
    inputs.push_back(MakeTensor(buffers[6], {72}, ge::DT_UINT8));
  } else {
    for (size_t index = 0; index < buffers.size(); ++index) {
      inputs.push_back(MakeTensor(buffers[index], kInputSpecs[index].shape,
                                  kInputSpecs[index].dtype));
    }
  }
  inputs.push_back(
      MakeTensor(control_bytes, {kControlInputElements}, ge::DT_INT32));
  ge::DataFlowInfo flow_info;
  const auto wall_start = std::chrono::steady_clock::now();
  *output_commit_state = CRUISE_EPOCH_EXECUTING;
  auto ret = engine->session->FeedDataFlowGraph(
      0, inputs, flow_info, kFeedTimeoutMs);
  *output_feed_calls = 1;
  if (ret != ge::SUCCESS) return 30;
  std::vector<ge::Tensor> outputs;
  ret = engine->session->FetchDataFlowGraph(
      0, outputs, flow_info, kFetchTimeoutMs);
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
      reinterpret_cast<const int64_t *>(outputs[0].GetData());
  const auto *result_control =
      reinterpret_cast<const int32_t *>(outputs[1].GetData());
  *output_device_status = result_control[3];
  *output_model_calls = result_control[4];
  *output_kv_import_checksum = result_control[5];
  if (*output_device_status == 0) {
    if (direct_device_import && *output_kv_import_checksum == 0) return 34;
    if (importing && !direct_device_import &&
        static_cast<uint32_t>(*output_kv_import_checksum) !=
            expected_import_checksum) {
      return 34;
    }
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
    for (const auto &entry : engine->ipc_imports) {
      aclrtIpcMemClose(entry.first.c_str());
    }
    engine->ipc_imports.clear();
    if (engine->device_import_payload != nullptr) {
      aclrtFree(engine->device_import_payload);
      engine->device_import_payload = nullptr;
    }
    engine->session.reset();
    ge::GEFinalize();
  }
  delete engine;
  g_engine_active = false;
}
