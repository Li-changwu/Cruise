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
constexpr size_t kCacheElements = 4096U;
constexpr size_t kMetadataElements = 4U;
constexpr size_t kReportElements = 9U;
constexpr size_t kSummaryElements = 22U;
constexpr int32_t kSlot = 2048;
constexpr uint16_t kLeftSentinel = 0x3e00U;
constexpr uint16_t kRightSentinel = 0xbe00U;
constexpr uint16_t kFirstValue = 0x3f80U;
constexpr uint16_t kSecondValue = 0x4000U;

ge::Tensor MakeTensor(const std::vector<int64_t> &shape, ge::DataType dtype,
                      const void *data, size_t bytes) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype));
  tensor.SetData(reinterpret_cast<const uint8_t *>(data), bytes);
  return tensor;
}

ge::Tensor MakeMetadata(int32_t sequence, uint16_t expected,
                        uint16_t requested) {
  std::array<int32_t, kMetadataElements> values = {
      sequence, kSlot, static_cast<int32_t>(expected),
      static_cast<int32_t>(requested)};
  return MakeTensor({static_cast<int64_t>(kMetadataElements)}, ge::DT_INT32,
                    values.data(), sizeof(values));
}

ge::Graph LoadAir(const std::string &air_path, uint32_t *load_status = nullptr,
                  bool *valid = nullptr) {
  ge::Graph graph("CruiseV2DeviceKvUpdateProbe");
  const auto status = graph.LoadFromFile(air_path.c_str());
  if (load_status != nullptr) *load_status = static_cast<uint32_t>(status);
  if (valid != nullptr) *valid = graph.IsValid();
  std::cout << "V2_DEVICE_KV_UPDATE_AIR_LOAD status=" << status
            << " valid=" << graph.IsValid() << std::endl;
  return graph;
}

ge::dflow::FlowGraph BuildFlowGraph(const std::string &air_path,
                                    const std::string &graph_config,
                                    const std::string &function_config) {
  using namespace ge::dflow;
  auto metadata = FlowData("metadata", 0);
  auto update = GraphPp("device_kv_update_graph_pp",
                        [air_path]() { return LoadAir(air_path); })
                    .SetCompileConfig(graph_config.c_str());
  auto controller = FunctionPp("device_kv_update_controller_pp")
                        .SetCompileConfig(function_config.c_str());
  controller.AddInvokedClosure("kv_update_graph_0", update);
  auto node = FlowNode("device_kv_update_controller_node", 1, 1);
  node.AddPp(controller).SetInput(0, metadata);
  FlowGraph graph("cruise_v2_device_kv_update_dataflow");
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
                 int32_t sequence, uint16_t expected, uint16_t requested) {
  return report[0] == sequence && report[1] == kSlot &&
         report[2] == expected && report[3] == requested &&
         report[4] == expected && report[5] == requested &&
         report[6] == kLeftSentinel && report[7] == kRightSentinel &&
         report[8] == 0;
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
                  int32_t sequence, uint16_t expected, uint16_t requested) {
  return value[0] == sequence && value[1] == 0 && value[2] == 1 &&
         value[3] == sequence && value[4] == 1 && value[5] == 1 &&
         value[6] == 0 && value[7] == 1 && value[8] == kSlot &&
         value[9] == expected && value[10] == requested &&
         value[11] == expected && value[12] == requested &&
         value[13] == kLeftSentinel && value[14] == kRightSentinel &&
         value[15] == 1 && value[16] == kCacheElements * sizeof(uint16_t) &&
         value[17] == 0 && value[18] == 0 &&
         value[19] == kReportElements;
}

void WriteArray(std::ofstream &stream,
                const std::array<int64_t, kSummaryElements> &values) {
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
         << "  \"gate\": \"V2-DEVICE-KV-UPDATE-GRAPH\",\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"status\": " << static_cast<uint32_t>(status) << ",\n"
         << "  \"add_status\": " << static_cast<uint32_t>(add_status)
         << ",\n"
         << "  \"model_load_status\": " << load_status << ",\n"
         << "  \"model_valid\": " << (model_valid ? "true" : "false")
         << ",\n"
         << "  \"report_exact\": " << (exact ? "true" : "false")
         << ",\n"
         << "  \"report\": [";
  for (size_t index = 0; index < report.size(); ++index) {
    if (index != 0) stream << ", ";
    stream << report[index];
  }
  stream << "],\n"
         << "  \"synthetic_probe_cache_input_bytes\": "
         << kCacheElements * sizeof(uint16_t) << ",\n"
         << "  \"full_cache_output_bytes\": 0,\n"
         << "  \"report_output_bytes\": " << sizeof(report) << ",\n"
         << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
         << "  \"claim_boundary\": \"Ordinary Graph custom-kernel compatibility on an 8 KiB synthetic buffer only.\"\n"
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
         << "  \"gate\": \"V2-DEVICE-KV-UPDATE-GRAPHPP\",\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"status\": " << static_cast<uint32_t>(status) << ",\n"
         << "  \"add_status\": " << static_cast<uint32_t>(add_status)
         << ",\n"
         << "  \"compile_status\": "
         << static_cast<uint32_t>(compile_status) << ",\n"
         << "  \"first_feed_status\": " << static_cast<uint32_t>(first_feed)
         << ",\n"
         << "  \"first_fetch_status\": "
         << static_cast<uint32_t>(first_fetch) << ",\n"
         << "  \"second_feed_status\": "
         << static_cast<uint32_t>(second_feed) << ",\n"
         << "  \"second_fetch_status\": "
         << static_cast<uint32_t>(second_fetch) << ",\n"
         << "  \"model_load_status\": " << load_status << ",\n"
         << "  \"model_valid\": " << (model_valid ? "true" : "false")
         << ",\n"
         << "  \"exact_two_updates\": " << (exact ? "true" : "false")
         << ",\n"
         << "  \"allocation_count\": " << second[2] << ",\n"
         << "  \"graph_call_count\": " << second[3] << ",\n"
         << "  \"buffer_address_stable\": "
         << (first[4] == 1 && second[4] == 1 ? "true" : "false") << ",\n"
         << "  \"flowmsg_identity_stable\": "
         << (first[5] == 1 && second[5] == 1 ? "true" : "false") << ",\n"
         << "  \"cross_call_checksum_continuity\": "
         << (first[21] == second[20] ? "true" : "false") << ",\n"
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
         << "  \"claim_boundary\": \"Two exact mutations of one FunctionPp-owned synthetic Device buffer; no target KV, full Decoder, Device-copy, or P5 performance claim.\"\n"
         << "}\n";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: device_kv_update_probe_host MODE AIR GRAPH_CONFIG "
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
    std::vector<uint16_t> cache(kCacheElements, 0);
    cache[kSlot - 1] = kLeftSentinel;
    cache[kSlot + 1] = kRightSentinel;
    auto metadata = MakeMetadata(1, 0, kFirstValue);
    std::vector<ge::Tensor> inputs = {
        MakeTensor({static_cast<int64_t>(kCacheElements)}, ge::DT_BF16,
                   cache.data(), cache.size() * sizeof(uint16_t)),
        metadata,
    };
    std::vector<ge::Tensor> outputs;
    add_status = session->AddGraph(kGraphId, source_graph);
    status = add_status;
    if (status == ge::SUCCESS) status = session->RunGraph(kGraphId, inputs, outputs);
    exact = status == ge::SUCCESS && ParseReport(outputs, report) &&
            ExactReport(report, 1, 0, kFirstValue);
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
          kGraphId, {MakeMetadata(1, 0, kFirstValue)}, flow_info,
          kFeedTimeoutMs);
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
        ExactSummary(first, 1, 0, kFirstValue);
    outputs.clear();
    if (status == ge::SUCCESS && first_exact) {
      second_feed = session->FeedDataFlowGraph(
          kGraphId, {MakeMetadata(2, kFirstValue, kSecondValue)}, flow_info,
          kFeedTimeoutMs);
      status = second_feed;
    }
    if (status == ge::SUCCESS && first_exact) {
      second_fetch = session->FetchDataFlowGraph(
          kGraphId, outputs, flow_info, kFetchTimeoutMs);
      status = second_fetch;
    }
    const bool second_exact =
        status == ge::SUCCESS && ParseSummary(outputs, second) &&
        ExactSummary(second, 2, kFirstValue, kSecondValue);
    exact = first_exact && second_exact && first[21] == second[20];
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started)
            .count();
    WriteDataflowSummary(summary_path, status, add_status, compile_status,
                         first_feed, first_fetch, second_feed, second_fetch,
                         load_status, model_valid, first, second, exact,
                         elapsed_ms);
  }

  std::cout << "V2_DEVICE_KV_UPDATE_RESULT mode=" << mode
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
