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

#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "graph/graph.h"

namespace {
constexpr uint32_t kGraphId = 0;
constexpr int32_t kFeedTimeoutMs = 120000;
constexpr int32_t kFetchTimeoutMs = 120000;
constexpr size_t kBatchSize = 4U;
constexpr size_t kBlocksPerRow = 3U;
constexpr size_t kPhysicalBlocks = 12U;
constexpr size_t kPackedChannels = 32U;
constexpr size_t kBlockSize = 128U;
constexpr size_t kPackWidth = 16U;
constexpr size_t kFeatures = kPackedChannels * kPackWidth;
constexpr size_t kStateElements =
    2U * kPhysicalBlocks * kPackedChannels * kBlockSize * kPackWidth;
constexpr size_t kMetadataElements = 5U;
constexpr size_t kReportElements = 18U;
constexpr size_t kSummaryElements = 24U;
constexpr size_t kUpdatedElements = 2U * kBatchSize * kFeatures;
constexpr uint32_t kChecksumModulus = 65521U;
constexpr std::array<int32_t, kBatchSize> kPositions = {0, 127, 128, 383};

uint16_t ExpectedBits(int32_t sequence, size_t cache_kind, size_t row,
                      size_t feature) {
  return static_cast<uint16_t>(0x2000U +
                               static_cast<uint32_t>(sequence) * 0x1000U +
                               static_cast<uint32_t>(cache_kind) * 0x0800U +
                               static_cast<uint32_t>(row) * 0x0200U +
                               static_cast<uint32_t>(feature));
}

uint32_t UpdatedChecksum(int32_t sequence) {
  uint32_t checksum = 0;
  for (size_t cache_kind = 0; cache_kind < 2U; ++cache_kind) {
    for (size_t row = 0; row < kBatchSize; ++row) {
      for (size_t feature = 0; feature < kFeatures; ++feature) {
        checksum =
            (checksum + ExpectedBits(sequence, cache_kind, row, feature)) %
            kChecksumModulus;
      }
    }
  }
  return checksum;
}

ge::Tensor MakeTensor(const std::vector<int64_t> &shape, ge::DataType dtype,
                      const void *data, size_t bytes) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype));
  tensor.SetData(reinterpret_cast<const uint8_t *>(data), bytes);
  return tensor;
}

ge::Tensor MakeMetadata(int32_t sequence) {
  std::array<int32_t, kMetadataElements> values = {
      sequence, kPositions[0], kPositions[1], kPositions[2], kPositions[3]};
  return MakeTensor({static_cast<int64_t>(kMetadataElements)}, ge::DT_INT32,
                    values.data(), sizeof(values));
}

ge::Graph LoadAir(const std::string &air_path, uint32_t *load_status = nullptr,
                  bool *valid = nullptr) {
  ge::Graph graph("CruiseV2PagedKvOrderProbe");
  const auto status = graph.LoadFromFile(air_path.c_str());
  if (load_status != nullptr) *load_status = static_cast<uint32_t>(status);
  if (valid != nullptr) *valid = graph.IsValid();
  std::cout << "V2_KV_ORDER_AIR_LOAD status=" << status
            << " valid=" << graph.IsValid() << std::endl;
  return graph;
}

ge::dflow::FlowGraph BuildFlowGraph(const std::string &air_path,
                                    const std::string &graph_config,
                                    const std::string &function_config) {
  using namespace ge::dflow;
  auto metadata = FlowData("metadata", 0);
  auto order = GraphPp("paged_kv_order_graph_pp",
                       [air_path]() { return LoadAir(air_path); })
                   .SetCompileConfig(graph_config.c_str());
  auto controller = FunctionPp("paged_kv_order_controller_pp")
                        .SetCompileConfig(function_config.c_str());
  controller.AddInvokedClosure("paged_kv_order_graph_0", order);
  auto node = FlowNode("paged_kv_order_controller_node", 1, 1);
  node.AddPp(controller).SetInput(0, metadata);
  FlowGraph graph("cruise_v2_paged_kv_order_dataflow");
  graph.SetInputs({metadata})
      .SetOutputs({node})
      .SetContainsNMappingNode(true);
  return graph;
}

bool ParseReport(const std::vector<ge::Tensor> &outputs,
                 std::array<int32_t, kReportElements> &report) {
  if (outputs.size() != 1 || outputs[0].GetData() == nullptr ||
      outputs[0].GetSize() != sizeof(report)) {
    return false;
  }
  std::memcpy(report.data(), outputs[0].GetData(), sizeof(report));
  return true;
}

bool ExactReport(const std::array<int32_t, kReportElements> &report,
                 int32_t sequence) {
  const uint32_t checksum = UpdatedChecksum(sequence);
  return report[0] == sequence && report[1] == sequence && report[2] == 0 &&
         report[3] == static_cast<int32_t>(kUpdatedElements) &&
         report[4] == static_cast<int32_t>(kUpdatedElements) &&
         report[5] == 0 && report[6] == static_cast<int32_t>(checksum) &&
         report[7] == static_cast<int32_t>(checksum) && report[8] == 0 &&
         report[9] == 511 && report[10] == 896 && report[11] == 1535 &&
         report[12] == ExpectedBits(sequence, 0, 0, 0) &&
         report[13] == ExpectedBits(sequence, 0, 3, kFeatures - 1U) &&
         report[14] == ExpectedBits(sequence, 1, 0, 0) &&
         report[15] == ExpectedBits(sequence, 1, 3, kFeatures - 1U) &&
         report[16] == 0 && report[17] == 0;
}

bool ParseSummary(const std::vector<ge::Tensor> &outputs,
                  std::array<int64_t, kSummaryElements> &summary) {
  if (outputs.size() != 1 || outputs[0].GetData() == nullptr ||
      outputs[0].GetSize() != sizeof(summary)) {
    return false;
  }
  std::memcpy(summary.data(), outputs[0].GetData(), sizeof(summary));
  return true;
}

bool ExactSummary(const std::array<int64_t, kSummaryElements> &value,
                  int32_t sequence) {
  const uint16_t before =
      sequence == 1 ? 0U : ExpectedBits(sequence - 1, 0, 0, 0);
  const uint32_t checksum = UpdatedChecksum(sequence);
  return value[0] == sequence && value[1] == 0 && value[2] == 1 &&
         value[3] == sequence && value[4] == 1 && value[5] == 1 &&
         value[6] == 0 && value[7] == 1 && value[8] == 1 &&
         value[9] == static_cast<int64_t>(kStateElements) &&
         value[10] == static_cast<int64_t>(kStateElements * sizeof(uint16_t)) &&
         value[11] == 0 && value[12] == 0 &&
         value[13] == static_cast<int64_t>(kReportElements) &&
         value[16] == before && value[17] == ExpectedBits(sequence, 0, 0, 0) &&
         value[18] == 0 && value[19] == checksum && value[20] == checksum &&
         value[21] == 0 && value[22] == 0 && value[23] == checksum;
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

void WriteGraphSummary(const std::string &path, ge::Status status,
                       ge::Status add_status, uint32_t load_status,
                       bool model_valid, bool exact,
                       const std::array<int32_t, kReportElements> &report,
                       int64_t elapsed_ms) {
  const bool pass = status == ge::SUCCESS && add_status == ge::SUCCESS &&
                    load_status == ge::GRAPH_SUCCESS && model_valid && exact;
  std::ofstream stream(path, std::ios::trunc);
  stream << "{\n"
         << "  \"gate\": \"V2-KV-ORDER-GRAPH\",\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"status\": " << static_cast<uint32_t>(status) << ",\n"
         << "  \"add_status\": " << static_cast<uint32_t>(add_status)
         << ",\n"
         << "  \"model_load_status\": " << load_status << ",\n"
         << "  \"model_valid\": " << (model_valid ? "true" : "false")
         << ",\n"
         << "  \"report_exact\": " << (exact ? "true" : "false")
         << ",\n"
         << "  \"explicit_dependency_observed\": "
         << (report[16] == 0 ? "true" : "false") << ",\n"
         << "  \"report\": ";
  WriteArray(stream, report);
  stream << ",\n"
         << "  \"ordinary_graph_state_input_bytes\": "
         << kStateElements * sizeof(uint16_t) << ",\n"
         << "  \"full_state_output_bytes\": 0,\n"
         << "  \"report_output_bytes\": " << sizeof(report) << ",\n"
         << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
         << "  \"claim_boundary\": \"Ordinary Graph compatibility and exact dependent reader report only.\"\n"
         << "}\n";
}

void WriteDataflowSummary(
    const std::string &path, ge::Status status, ge::Status add_status,
    ge::Status compile_status, ge::Status first_feed, ge::Status first_fetch,
    ge::Status second_feed, ge::Status second_fetch, uint32_t load_status,
    bool model_valid, const std::array<int64_t, kSummaryElements> &first,
    const std::array<int64_t, kSummaryElements> &second, bool exact,
    int64_t elapsed_ms) {
  const bool pass = status == ge::SUCCESS && add_status == ge::SUCCESS &&
                    compile_status == ge::SUCCESS &&
                    first_feed == ge::SUCCESS && first_fetch == ge::SUCCESS &&
                    second_feed == ge::SUCCESS &&
                    second_fetch == ge::SUCCESS &&
                    load_status == ge::GRAPH_SUCCESS && model_valid && exact;
  std::ofstream stream(path, std::ios::trunc);
  stream << "{\n"
         << "  \"gate\": \"V2-KV-ORDER-GRAPHPP\",\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"status\": " << static_cast<uint32_t>(status) << ",\n"
         << "  \"add_status\": " << static_cast<uint32_t>(add_status)
         << ",\n"
         << "  \"compile_status\": " << static_cast<uint32_t>(compile_status)
         << ",\n"
         << "  \"first_feed_status\": " << static_cast<uint32_t>(first_feed)
         << ",\n"
         << "  \"first_fetch_status\": " << static_cast<uint32_t>(first_fetch)
         << ",\n"
         << "  \"second_feed_status\": " << static_cast<uint32_t>(second_feed)
         << ",\n"
         << "  \"second_fetch_status\": " << static_cast<uint32_t>(second_fetch)
         << ",\n"
         << "  \"model_load_status\": " << load_status << ",\n"
         << "  \"model_valid\": " << (model_valid ? "true" : "false")
         << ",\n"
         << "  \"exact_two_ordered_updates\": "
         << (exact ? "true" : "false") << ",\n"
         << "  \"allocation_count\": " << second[2] << ",\n"
         << "  \"graph_call_count\": " << second[3] << ",\n"
         << "  \"buffer_address_stable\": "
         << (first[4] == 1 && second[4] == 1 ? "true" : "false") << ",\n"
         << "  \"flowmsg_identity_stable\": "
         << (first[5] == 1 && second[5] == 1 ? "true" : "false") << ",\n"
         << "  \"cross_call_checksum_continuity\": "
         << (first[15] == second[14] ? "true" : "false") << ",\n"
         << "  \"full_state_exact_after_each_call\": "
         << (first[8] == 1 && second[8] == 1 ? "true" : "false") << ",\n"
         << "  \"explicit_dependency_observed_each_call\": "
         << (first[21] == 0 && second[21] == 0 ? "true" : "false")
         << ",\n"
         << "  \"device_owned_state_bytes\": " << second[10] << ",\n"
         << "  \"host_cache_input_bytes\": 0,\n"
         << "  \"host_cache_output_bytes\": 0,\n"
         << "  \"host_trigger_input_bytes\": "
         << 2 * kMetadataElements * sizeof(int32_t) << ",\n"
         << "  \"host_summary_output_bytes\": "
         << 2 * kSummaryElements * sizeof(int64_t) << ",\n"
         << "  \"raw_device_address_abi_used\": false,\n"
         << "  \"external_refdata_used\": false,\n"
         << "  \"first_summary\": ";
  WriteArray(stream, first);
  stream << ",\n  \"second_summary\": ";
  WriteArray(stream, second);
  stream << ",\n"
         << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
         << "  \"claim_boundary\": \"Two exact ordered updates of one FunctionPp-owned B4/K384 state tensor; no attention, full Decoder, zero-copy, or P5 claim.\"\n"
         << "}\n";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: paged_kv_order_probe_host MODE AIR GRAPH_CONFIG "
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
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
  };
  if (mode == "dataflow") {
    config["ge.exec.logicalDeviceClusterDeployMode"] = "SINGLE";
    config["ge.exec.logicalDeviceId"] = "[0:0]";
    config["ge.experiment.data_flow_deploy_info_path"] = deploy_config.c_str();
  }

  auto status = ge::GEInitialize(config);
  ge::Status add_status = ge::FAILED;
  ge::Status compile_status = ge::FAILED;
  ge::Status first_feed = ge::FAILED;
  ge::Status first_fetch = ge::FAILED;
  ge::Status second_feed = ge::FAILED;
  ge::Status second_fetch = ge::FAILED;
  uint32_t load_status = UINT32_MAX;
  bool model_valid = false;
  std::array<int32_t, kReportElements> report{};
  std::array<int64_t, kSummaryElements> first{};
  std::array<int64_t, kSummaryElements> second{};
  if (status != ge::SUCCESS) return 10;

  auto session = std::make_shared<ge::Session>(config);
  const auto started = std::chrono::steady_clock::now();
  auto source_graph = LoadAir(air_path, &load_status, &model_valid);
  bool exact = false;
  if (mode == "graph") {
    std::vector<uint16_t> state(kStateElements, 0);
    std::vector<ge::Tensor> inputs = {
        MakeTensor({2, 12, 32, 128, 16}, ge::DT_BF16, state.data(),
                   state.size() * sizeof(uint16_t)),
        MakeMetadata(1),
    };
    std::vector<ge::Tensor> outputs;
    add_status = session->AddGraph(kGraphId, source_graph);
    status = add_status;
    if (status == ge::SUCCESS) status = session->RunGraph(kGraphId, inputs, outputs);
    exact = status == ge::SUCCESS && ParseReport(outputs, report) &&
            ExactReport(report, 1);
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started)
            .count();
    WriteGraphSummary(summary_path, status, add_status, load_status, model_valid,
                      exact, report, elapsed_ms);
  } else {
    auto flow_graph = BuildFlowGraph(air_path, graph_config, function_config);
    add_status = session->AddGraph(kGraphId, flow_graph.ToGeGraph());
    status = add_status;
    if (status == ge::SUCCESS) {
      compile_status = session->CompileGraph(kGraphId);
      status = compile_status;
    }
    ge::DataFlowInfo flow_info;
    if (status == ge::SUCCESS) {
      first_feed = session->FeedDataFlowGraph(
          kGraphId, {MakeMetadata(1)}, flow_info, kFeedTimeoutMs);
      status = first_feed;
    }
    std::vector<ge::Tensor> outputs;
    if (status == ge::SUCCESS) {
      first_fetch = session->FetchDataFlowGraph(
          kGraphId, outputs, flow_info, kFetchTimeoutMs);
      status = first_fetch;
    }
    const bool first_exact =
        status == ge::SUCCESS && ParseSummary(outputs, first) &&
        ExactSummary(first, 1);
    outputs.clear();
    if (status == ge::SUCCESS && first_exact) {
      second_feed = session->FeedDataFlowGraph(
          kGraphId, {MakeMetadata(2)}, flow_info, kFeedTimeoutMs);
      status = second_feed;
    }
    if (status == ge::SUCCESS && first_exact) {
      second_fetch = session->FetchDataFlowGraph(
          kGraphId, outputs, flow_info, kFetchTimeoutMs);
      status = second_fetch;
    }
    const bool second_exact =
        status == ge::SUCCESS && ParseSummary(outputs, second) &&
        ExactSummary(second, 2);
    exact = first_exact && second_exact && first[15] == second[14];
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started)
            .count();
    WriteDataflowSummary(summary_path, status, add_status, compile_status,
                         first_feed, first_fetch, second_feed, second_fetch,
                         load_status, model_valid, first, second, exact,
                         elapsed_ms);
  }

  std::cout << "V2_KV_ORDER_RESULT mode=" << mode
            << " status=" << static_cast<uint32_t>(status)
            << " exact=" << exact << std::endl;
  if (add_status == ge::SUCCESS) session->RemoveGraph(kGraphId);
  session.reset();
  ge::GEFinalize();
  return status == ge::SUCCESS && load_status == ge::GRAPH_SUCCESS &&
                 model_valid && exact
             ? 0
             : 10;
}
