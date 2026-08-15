#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include "acl/acl.h"
#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "graph/graph.h"

namespace {
constexpr uint32_t kGraphId = 0;
constexpr int32_t kFeedTimeoutMs = 240000;
constexpr int32_t kFetchTimeoutMs = 240000;
constexpr size_t kCacheElements = 12U * 4U * 8U * 128U * 16U;
constexpr size_t kQueryElements = 4U * 28U * 128U;
constexpr size_t kMaskElements = 4U * 384U;
constexpr size_t kMetadataElements = 5U;
constexpr size_t kTicketElements = 9U;
constexpr size_t kSummaryElements = 32U;
constexpr std::array<int32_t, 4> kPositions = {0, 127, 128, 383};

uint16_t SequenceBits(int32_t sequence) {
  return sequence == 1 ? 0x3f80U : 0x4000U;
}

ge::Tensor MakeTensor(const std::vector<int64_t> &shape, ge::DataType dtype,
                      const void *data, size_t bytes) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype));
  tensor.SetData(reinterpret_cast<const uint8_t *>(data), bytes);
  return tensor;
}

ge::Tensor MakeMetadata(int32_t sequence) {
  const std::array<int32_t, kMetadataElements> values = {
      sequence, kPositions[0], kPositions[1], kPositions[2], kPositions[3]};
  return MakeTensor({5}, ge::DT_INT32, values.data(), sizeof(values));
}

ge::Graph LoadAir(const std::string &air_path, uint32_t *load_status = nullptr,
                  bool *valid = nullptr) {
  ge::Graph graph("CruiseV2AttentionKvProbe");
  const auto status = graph.LoadFromFile(air_path.c_str());
  if (load_status != nullptr) *load_status = static_cast<uint32_t>(status);
  if (valid != nullptr) *valid = graph.IsValid();
  std::cout << "V2_KV_ATTENTION_AIR_LOAD status=" << status
            << " valid=" << graph.IsValid() << std::endl;
  return graph;
}

ge::dflow::FlowGraph BuildFlowGraph(const std::string &air_path,
                                    const std::string &graph_config,
                                    const std::string &function_config) {
  using namespace ge::dflow;
  auto metadata = FlowData("metadata", 0);
  auto graph_pp = GraphPp("attention_kv_graph_pp",
                          [air_path]() { return LoadAir(air_path); })
                      .SetCompileConfig(graph_config.c_str());
  auto controller = FunctionPp("attention_kv_controller_pp")
                        .SetCompileConfig(function_config.c_str());
  controller.AddInvokedClosure("attention_kv_graph_0", graph_pp);
  auto node = FlowNode("attention_kv_controller_node", 1, 1);
  node.AddPp(controller).SetInput(0, metadata);
  FlowGraph graph("cruise_v2_attention_kv_dataflow");
  graph.SetInputs({metadata}).SetOutputs({node}).SetContainsNMappingNode(true);
  return graph;
}

class DeviceInputs {
 public:
  ~DeviceInputs() {
    tensors_.clear();
    for (void *pointer : pointers_) {
      if (pointer != nullptr) aclrtFree(pointer);
    }
  }

  bool Build(int32_t sequence) {
    std::vector<uint16_t> query(kQueryElements, 0x3f80U);
    std::array<int32_t, kMetadataElements> metadata = {
        sequence, kPositions[0], kPositions[1], kPositions[2], kPositions[3]};
    std::vector<uint8_t> mask(kMaskElements, 1U);
    for (size_t row = 0; row < kPositions.size(); ++row) {
      mask[row * 384U + static_cast<size_t>(kPositions[row])] = 0U;
    }
    std::array<int32_t, 12> block_table{};
    for (size_t index = 0; index < block_table.size(); ++index) {
      block_table[index] = static_cast<int32_t>(index);
    }
    return Add({12, 4, 8, 128, 16}, ge::DT_BF16, nullptr,
               kCacheElements * sizeof(uint16_t)) &&
           Add({12, 4, 8, 128, 16}, ge::DT_BF16, nullptr,
               kCacheElements * sizeof(uint16_t)) &&
           Add({4, 28, 1, 128}, ge::DT_BF16, query.data(),
               query.size() * sizeof(query[0])) &&
           Add({5}, ge::DT_INT32, metadata.data(), sizeof(metadata)) &&
           Add({4, 1, 1, 384}, ge::DT_BOOL, mask.data(), mask.size()) &&
           Add({4, 3}, ge::DT_INT32, block_table.data(), sizeof(block_table));
  }

  const std::vector<ge::Tensor> &tensors() const { return tensors_; }

 private:
  bool Add(const std::vector<int64_t> &shape, ge::DataType dtype,
           const void *host_data, size_t bytes) {
    void *device = nullptr;
    if (aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      return false;
    }
    pointers_.push_back(device);
    const aclError init =
        host_data == nullptr
            ? aclrtMemset(device, bytes, 0, bytes)
            : aclrtMemcpy(device, bytes, host_data, bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE);
    if (init != ACL_SUCCESS) return false;
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementDevice);
    ge::Tensor tensor;
    if (tensor.SetTensorDesc(desc) != ge::GRAPH_SUCCESS ||
        tensor.SetData(reinterpret_cast<uint8_t *>(device), bytes,
                       [](uint8_t *) {}) != ge::GRAPH_SUCCESS ||
        tensor.SetPlacement(ge::kPlacementDevice) != ge::GRAPH_SUCCESS) {
      return false;
    }
    tensors_.push_back(std::move(tensor));
    return true;
  }

  std::vector<void *> pointers_;
  std::vector<ge::Tensor> tensors_;
};

bool ParseGraphOutputs(const std::vector<ge::Tensor> &outputs, int32_t sequence,
                       bool *attention_exact, bool *ticket_exact) {
  if (outputs.size() != 2 || outputs[0].GetData() == nullptr ||
      outputs[1].GetData() == nullptr ||
      outputs[0].GetSize() != kQueryElements * sizeof(uint16_t) ||
      outputs[1].GetSize() != kTicketElements * sizeof(int32_t)) {
    return false;
  }
  const auto *attention =
      reinterpret_cast<const uint16_t *>(outputs[0].GetData());
  const auto *ticket = reinterpret_cast<const int32_t *>(outputs[1].GetData());
  *attention_exact = std::all_of(
      attention, attention + kQueryElements,
      [sequence](uint16_t value) { return value == SequenceBits(sequence); });
  *ticket_exact =
      ticket[0] == sequence && ticket[1] == 0 && ticket[2] == 4096 &&
      ticket[3] == 0 && ticket[4] == 127 && ticket[5] == 128 &&
      ticket[6] == 383 && ticket[7] == SequenceBits(sequence) &&
      ticket[8] == SequenceBits(sequence);
  return true;
}

bool ParseSummary(const std::vector<ge::Tensor> &outputs,
                  std::array<int64_t, kSummaryElements> *summary) {
  if (outputs.size() != 1 || outputs[0].GetData() == nullptr ||
      outputs[0].GetSize() != sizeof(*summary)) {
    return false;
  }
  std::memcpy(summary->data(), outputs[0].GetData(), sizeof(*summary));
  return true;
}

bool ExactSummary(const std::array<int64_t, kSummaryElements> &value,
                  int32_t sequence) {
  return value[0] == sequence && value[1] == 0 && value[2] == 1 &&
         value[3] == sequence && value[4] == 1 && value[5] == 1 &&
         value[6] == 1 && value[7] == 1 && value[8] == 1 && value[9] == 1 &&
         value[10] == 2 * static_cast<int64_t>(kCacheElements) * 2 &&
         value[11] == 0 && value[12] == 0 &&
         value[13] == static_cast<int64_t>(kQueryElements) &&
         value[14] == static_cast<int64_t>(kTicketElements) &&
         value[15] == (sequence == 1 ? 0 : SequenceBits(sequence - 1)) &&
         value[16] == SequenceBits(sequence) &&
         value[17] == SequenceBits(sequence) &&
         value[18] == SequenceBits(sequence) &&
         value[19] == SequenceBits(sequence) && value[20] == 0 &&
         value[21] == 4096 && value[22] == 0 && value[23] == 0 &&
         value[24] == 6 && value[25] == 2 && value[26] == 20 &&
         value[27] == static_cast<int64_t>(sizeof(value)) &&
         value[28] == 1 && value[29] == 1 && value[30] == 1 && value[31] == 1;
}

template <typename T, size_t N>
void WriteArray(std::ofstream &stream, const std::array<T, N> &values) {
  stream << "[";
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) stream << ", ";
    stream << values[index];
  }
  stream << "]";
}

void WriteGraphResult(const std::string &path, ge::Status status,
                      ge::Status add_status, uint32_t load_status,
                      bool model_valid, bool inputs_ready,
                      bool attention_exact, bool ticket_exact,
                      int64_t elapsed_ms) {
  const bool pass = status == ge::SUCCESS && add_status == ge::SUCCESS &&
                    load_status == ge::GRAPH_SUCCESS && model_valid &&
                    inputs_ready && attention_exact && ticket_exact;
  std::ofstream stream(path, std::ios::trunc);
  stream << "{\n"
         << "  \"gate\": \"V2-KV-ATTENTION-GRAPH\",\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"status\": " << static_cast<uint32_t>(status) << ",\n"
         << "  \"add_status\": " << static_cast<uint32_t>(add_status) << ",\n"
         << "  \"model_load_status\": " << load_status << ",\n"
         << "  \"model_valid\": " << (model_valid ? "true" : "false") << ",\n"
         << "  \"device_placed_inputs_ready\": "
         << (inputs_ready ? "true" : "false") << ",\n"
         << "  \"attention_exact\": " << (attention_exact ? "true" : "false") << ",\n"
         << "  \"ticket_exact\": " << (ticket_exact ? "true" : "false") << ",\n"
         << "  \"host_cache_input_bytes\": 0,\n"
         << "  \"host_cache_output_bytes\": 0,\n"
         << "  \"device_placed_cache_input_bytes\": " << 2 * kCacheElements * 2 << ",\n"
         << "  \"compact_output_bytes\": "
         << kQueryElements * 2 + kTicketElements * 4 << ",\n"
         << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
         << "  \"claim_boundary\": \"Ordinary Graph exact update-to-FIA compatibility only.\"\n"
         << "}\n";
}

void WriteDataflowResult(
    const std::string &path, ge::Status status, ge::Status add_status,
    ge::Status compile_status, ge::Status first_feed, ge::Status first_fetch,
    ge::Status second_feed, ge::Status second_fetch, uint32_t load_status,
    bool model_valid, const std::array<int64_t, kSummaryElements> &first,
    const std::array<int64_t, kSummaryElements> &second, bool exact,
    int64_t elapsed_ms) {
  const bool pass = status == ge::SUCCESS && add_status == ge::SUCCESS &&
                    compile_status == ge::SUCCESS && first_feed == ge::SUCCESS &&
                    first_fetch == ge::SUCCESS && second_feed == ge::SUCCESS &&
                    second_fetch == ge::SUCCESS && load_status == ge::GRAPH_SUCCESS &&
                    model_valid && exact;
  std::ofstream stream(path, std::ios::trunc);
  stream << "{\n"
         << "  \"gate\": \"V2-KV-ATTENTION-GRAPHPP\",\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"status\": " << static_cast<uint32_t>(status) << ",\n"
         << "  \"add_status\": " << static_cast<uint32_t>(add_status) << ",\n"
         << "  \"compile_status\": " << static_cast<uint32_t>(compile_status) << ",\n"
         << "  \"first_feed_status\": " << static_cast<uint32_t>(first_feed) << ",\n"
         << "  \"first_fetch_status\": " << static_cast<uint32_t>(first_fetch) << ",\n"
         << "  \"second_feed_status\": " << static_cast<uint32_t>(second_feed) << ",\n"
         << "  \"second_fetch_status\": " << static_cast<uint32_t>(second_fetch) << ",\n"
         << "  \"model_load_status\": " << load_status << ",\n"
         << "  \"model_valid\": " << (model_valid ? "true" : "false") << ",\n"
         << "  \"exact_two_update_attention_calls\": " << (exact ? "true" : "false") << ",\n"
         << "  \"allocation_count\": " << second[2] << ",\n"
         << "  \"graph_call_count\": " << second[3] << ",\n"
         << "  \"flowmsg_identity_stable\": " << (second[4] == 1 ? "true" : "false") << ",\n"
         << "  \"buffer_address_stable\": " << (second[5] == 1 ? "true" : "false") << ",\n"
         << "  \"full_cache_exact_after_each_call\": "
         << (first[7] == 1 && second[7] == 1 ? "true" : "false") << ",\n"
         << "  \"attention_observed_update_each_call\": "
         << (first[8] == 1 && second[8] == 1 ? "true" : "false") << ",\n"
         << "  \"device_owned_cache_bytes\": " << second[10] << ",\n"
         << "  \"host_cache_input_bytes\": 0,\n"
         << "  \"host_cache_output_bytes\": 0,\n"
         << "  \"host_metadata_input_bytes\": 40,\n"
         << "  \"host_summary_output_bytes\": " << 2 * sizeof(first) << ",\n"
         << "  \"raw_device_address_abi_used\": false,\n"
         << "  \"external_refdata_used\": false,\n"
         << "  \"first_summary\": ";
  WriteArray(stream, first);
  stream << ",\n  \"second_summary\": ";
  WriteArray(stream, second);
  stream << ",\n  \"elapsed_ms\": " << elapsed_ms << ",\n"
         << "  \"claim_boundary\": \"Combined FunctionPp-owned PA-NZ update-to-FIA safety gate only.\"\n"
         << "}\n";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: attention_kv_probe_host MODE AIR GRAPH_CONFIG "
                 "FUNCTION_CONFIG DEPLOY_CONFIG SUMMARY_JSON"
              << std::endl;
    return 2;
  }
  const std::string mode = argv[1];
  const std::string air_path = argv[2];
  const std::string graph_config = argv[3];
  const std::string function_config = argv[4];
  const std::string deploy_config = argv[5];
  const std::string summary_path = argv[6];
  if ((mode != "graph" && mode != "dataflow") ||
      access(air_path.c_str(), R_OK) != 0 ||
      access(graph_config.c_str(), R_OK) != 0 ||
      access(function_config.c_str(), R_OK) != 0 ||
      access(deploy_config.c_str(), R_OK) != 0) {
    return 2;
  }

  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.externalWeight", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
  };
  if (mode == "dataflow") {
    config["ge.exec.logicalDeviceClusterDeployMode"] = "SINGLE";
    config["ge.exec.logicalDeviceId"] = "[0:0]";
    config["ge.experiment.data_flow_deploy_info_path"] = deploy_config.c_str();
  }
  ge::Status status = ge::GEInitialize(config);
  if (status != ge::SUCCESS) return 10;
  auto session = std::make_shared<ge::Session>(config);
  uint32_t load_status = UINT32_MAX;
  bool model_valid = false;
  const ge::Graph source = LoadAir(air_path, &load_status, &model_valid);
  ge::Status add_status = ge::FAILED;
  const auto started = std::chrono::steady_clock::now();

  if (mode == "graph") {
    add_status = session->AddGraph(kGraphId, source);
    DeviceInputs inputs;
    const bool inputs_ready = add_status == ge::SUCCESS && inputs.Build(1);
    std::vector<ge::Tensor> outputs;
    status = inputs_ready ? session->RunGraph(kGraphId, inputs.tensors(), outputs)
                          : ge::FAILED;
    bool attention_exact = false;
    bool ticket_exact = false;
    if (status == ge::SUCCESS) {
      ParseGraphOutputs(outputs, 1, &attention_exact, &ticket_exact);
    }
    const int64_t elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started)
            .count();
    WriteGraphResult(summary_path, status, add_status, load_status, model_valid,
                     inputs_ready, attention_exact, ticket_exact, elapsed);
    status = status == ge::SUCCESS && inputs_ready && attention_exact &&
                     ticket_exact
                 ? ge::SUCCESS
                 : ge::FAILED;
  } else {
    const auto flow = BuildFlowGraph(air_path, graph_config, function_config);
    add_status = session->AddGraph(kGraphId, flow.ToGeGraph());
    ge::Status compile_status = add_status == ge::SUCCESS
                                    ? session->CompileGraph(kGraphId)
                                    : ge::FAILED;
    std::array<int64_t, kSummaryElements> first{};
    std::array<int64_t, kSummaryElements> second{};
    ge::Status feeds[2] = {ge::FAILED, ge::FAILED};
    ge::Status fetches[2] = {ge::FAILED, ge::FAILED};
    for (int32_t sequence = 1; sequence <= 2 && compile_status == ge::SUCCESS;
         ++sequence) {
      ge::DataFlowInfo info;
      feeds[sequence - 1] = session->FeedDataFlowGraph(
          kGraphId, {MakeMetadata(sequence)}, info, kFeedTimeoutMs);
      std::vector<ge::Tensor> outputs;
      if (feeds[sequence - 1] == ge::SUCCESS) {
        fetches[sequence - 1] = session->FetchDataFlowGraph(
            kGraphId, outputs, info, kFetchTimeoutMs);
      }
      if (fetches[sequence - 1] == ge::SUCCESS) {
        ParseSummary(outputs, sequence == 1 ? &first : &second);
      }
    }
    const bool exact = ExactSummary(first, 1) && ExactSummary(second, 2);
    status = exact && fetches[1] == ge::SUCCESS ? ge::SUCCESS : ge::FAILED;
    const int64_t elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started)
            .count();
    WriteDataflowResult(summary_path, status, add_status, compile_status,
                        feeds[0], fetches[0], feeds[1], fetches[1], load_status,
                        model_valid, first, second, exact, elapsed);
  }
  session->RemoveGraph(kGraphId);
  session.reset();
  ge::GEFinalize();
  return status == ge::SUCCESS && load_status == ge::GRAPH_SUCCESS && model_valid
             ? 0
             : 10;
}
