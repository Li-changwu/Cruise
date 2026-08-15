#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
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
constexpr size_t kQueryElements = 4 * 28 * 1 * 128;
constexpr size_t kKvElements = 4 * 4 * 384 * 128;
constexpr size_t kMaskElements = 4 * 1 * 1 * 384;
constexpr size_t kPhysicalBlocks = 12;
constexpr size_t kPaPackedChannels = 32;
constexpr size_t kBlockSize = 128;
constexpr size_t kPaPackWidth = 16;
constexpr size_t kBlockTableElements = 4 * 3;

ge::Tensor MakeBf16(const std::vector<int64_t> &shape, size_t elements) {
  std::vector<uint16_t> values(elements, 0x3f80);
  ge::Tensor tensor;
  tensor.SetTensorDesc(
      ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_BF16));
  tensor.SetData(reinterpret_cast<const uint8_t *>(values.data()),
                 values.size() * sizeof(values[0]));
  return tensor;
}

ge::Tensor MakeMask() {
  std::vector<uint8_t> values(kMaskElements, 0);
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape({4, 1, 1, 384}),
                                      ge::FORMAT_ND, ge::DT_BOOL));
  tensor.SetData(values.data(), values.size());
  return tensor;
}

ge::Tensor MakeBlockTable() {
  std::vector<int32_t> values(kBlockTableElements);
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<int32_t>(index);
  }
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape({4, 3}), ge::FORMAT_ND,
                                      ge::DT_INT32));
  tensor.SetData(reinterpret_cast<const uint8_t *>(values.data()),
                 values.size() * sizeof(values[0]));
  return tensor;
}

std::vector<ge::Tensor> MakeInputs(const std::string &kv_layout) {
  if (kv_layout == "pa-nz") {
    const size_t elements = kPhysicalBlocks * kPaPackedChannels *
                            kBlockSize * kPaPackWidth;
    return {
        MakeBf16({4, 28, 1, 128}, kQueryElements),
        MakeBf16({12, 32, 128, 16}, elements),
        MakeBf16({12, 32, 128, 16}, elements),
        MakeBlockTable(),
    };
  }
  return {
      MakeBf16({4, 28, 1, 128}, kQueryElements),
      MakeBf16({4, 4, 384, 128}, kKvElements),
      MakeBf16({4, 4, 384, 128}, kKvElements),
      MakeMask(),
  };
}

std::vector<uint8_t> ReadBinary(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
}

ge::Graph LoadAir(const std::string &air_path, uint32_t *load_status = nullptr,
                  bool *valid = nullptr) {
  ge::Graph graph("MiniFIAGraph");
  const auto status = graph.LoadFromFile(air_path.c_str());
  if (load_status != nullptr) *load_status = static_cast<uint32_t>(status);
  if (valid != nullptr) *valid = graph.IsValid();
  std::cout << "MINI_FIA_AIR_LOAD status=" << status
            << " valid=" << graph.IsValid() << std::endl;
  return graph;
}

ge::Graph LoadSerializedModel(const std::string &model_path,
                              uint32_t *load_status = nullptr,
                              bool *valid = nullptr) {
  auto model = ReadBinary(model_path);
  ge::Graph graph("MiniFIASerializedModel");
  const auto status = model.empty()
                          ? static_cast<ge::graphStatus>(1)
                          : graph.LoadFromSerializedModelArray(model.data(),
                                                               model.size());
  if (load_status != nullptr) *load_status = static_cast<uint32_t>(status);
  if (valid != nullptr) *valid = graph.IsValid();
  std::cout << "MINI_FIA_SERIALIZED_LOAD status=" << status
            << " valid=" << graph.IsValid() << " bytes=" << model.size()
            << std::endl;
  return graph;
}

ge::dflow::FlowGraph BuildFlowGraph(const std::string &model_path,
                                    const std::string &graph_config,
                                    bool serialized) {
  using namespace ge::dflow;
  auto query = FlowData("query", 0);
  auto key = FlowData("key", 1);
  auto value = FlowData("value", 2);
  auto mask = FlowData("mask", 3);
  auto fia = GraphPp(serialized ? "mini_fia_serialized_graph_pp"
                                : "mini_fia_graph_pp",
                     [model_path, serialized]() {
                       return serialized ? LoadSerializedModel(model_path)
                                         : LoadAir(model_path);
                     })
                 .SetCompileConfig(graph_config.c_str());
  auto node = FlowNode("mini_fia_node", 4, 1);
  node.AddPp(fia)
      .SetInput(0, query)
      .SetInput(1, key)
      .SetInput(2, value)
      .SetInput(3, mask);
  FlowGraph graph("mini_fia_dataflow");
  graph.SetInputs({query, key, value, mask}).SetOutputs({node});
  return graph;
}

bool OutputExact(const std::vector<ge::Tensor> &outputs) {
  if (outputs.size() != 1 || outputs[0].GetData() == nullptr ||
      outputs[0].GetSize() != kQueryElements * sizeof(uint16_t)) {
    return false;
  }
  const auto *values =
      reinterpret_cast<const uint16_t *>(outputs[0].GetData());
  return std::all_of(values, values + kQueryElements,
                     [](uint16_t value) { return value == 0x3f80; });
}

void WriteSummary(const std::string &path, const std::string &mode,
                  ge::Status status, bool exact, int64_t elapsed_ms,
                  uint32_t model_load_status, bool model_valid,
                  const std::string &jit_compile,
                  const std::string &kv_layout) {
  if (path.empty()) return;
  std::ofstream stream(path);
  stream << "{\n"
         << "  \"gate\": \""
         << (kv_layout == "pa-nz" ? "V2-PA-NZ-FIA-GRAPHPP"
                                    : "P3-MINI-FIA-GRAPHPP")
         << "\",\n"
         << "  \"pass\": "
         << (status == ge::SUCCESS && exact &&
                     model_load_status == ge::GRAPH_SUCCESS && model_valid
                 ? "true"
                 : "false")
         << ",\n"
         << "  \"mode\": \"" << mode << "\",\n"
         << "  \"kv_layout\": \"" << kv_layout << "\",\n"
         << "  \"page_attention\": "
         << (kv_layout == "pa-nz" ? "true" : "false") << ",\n"
         << "  \"status\": " << static_cast<uint32_t>(status) << ",\n"
         << "  \"model_load_status\": " << model_load_status << ",\n"
         << "  \"model_valid\": " << (model_valid ? "true" : "false")
         << ",\n"
         << "  \"ge_jit_compile\": \"" << jit_compile << "\",\n"
         << "  \"output_exact\": " << (exact ? "true" : "false") << ",\n"
         << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
         << "  \"claim_boundary\": \"Single weight-free FIA Graph or GraphPp load only; no P3 Owner claim.\"\n"
         << "}\n";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: mini_fia_graphpp_probe MODE AIR_OR_OM GRAPH_CONFIG "
                 "DEPLOY_CONFIG SUMMARY_JSON EMPTY_EXTERNAL_WEIGHT_DIR "
                 "KV_LAYOUT"
              << std::endl;
    return 2;
  }
  const std::string mode = argv[1];
  const std::string model_path = argv[2];
  const std::string graph_config = argv[3];
  const std::string deploy_config = argv[4];
  const std::string summary_path = argv[5];
  const std::string external_weight_dir = argv[6];
  const std::string kv_layout = argv[7];
  const char *jit_compile_env =
      std::getenv("CRUISE_MINI_FIA_GE_JIT_COMPILE");
  const std::string jit_compile =
      jit_compile_env == nullptr || jit_compile_env[0] == '\0'
          ? "default"
          : jit_compile_env;
  if ((mode != "graph" && mode != "dataflow" &&
       mode != "serialized-dataflow") ||
      (jit_compile != "default" && jit_compile != "0" &&
       jit_compile != "1") ||
      access(model_path.c_str(), R_OK) != 0 ||
      access(graph_config.c_str(), R_OK) != 0 ||
      access(deploy_config.c_str(), R_OK) != 0 ||
      access(external_weight_dir.c_str(), R_OK) != 0 ||
      (kv_layout != "dense" && kv_layout != "pa-nz")) {
    return 2;
  }

  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.externalWeight", "1"},
      {"ge.externalWeightDir", external_weight_dir.c_str()},
      {"ge.modelFileNamePrefix", "mini_fia_probe"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
  };
  if (jit_compile != "default") {
    config["ge.jit_compile"] = jit_compile.c_str();
  }
  if (mode == "dataflow") {
    config["ge.exec.logicalDeviceClusterDeployMode"] = "SINGLE";
    config["ge.exec.logicalDeviceId"] = "[0:0]";
    config["ge.experiment.data_flow_deploy_info_path"] = deploy_config.c_str();
  }
  auto status = ge::GEInitialize(config);
  uint32_t model_load_status = UINT32_MAX;
  bool model_valid = false;
  if (status != ge::SUCCESS) {
    WriteSummary(summary_path, mode, status, false, 0, model_load_status,
                 model_valid, jit_compile, kv_layout);
    std::cout << "MINI_FIA_INIT_FAIL mode=" << mode
              << " status=" << static_cast<uint32_t>(status) << std::endl;
    return 10;
  }
  auto session = std::make_shared<ge::Session>(config);
  const auto started = std::chrono::steady_clock::now();
  std::vector<ge::Tensor> outputs;
  ge::Graph source_graph = mode == "serialized-dataflow"
                               ? LoadSerializedModel(model_path,
                                                     &model_load_status,
                                                     &model_valid)
                               : LoadAir(model_path, &model_load_status,
                                         &model_valid);
  if (mode == "graph") {
    status = session->AddGraph(kGraphId, source_graph);
    if (status == ge::SUCCESS) {
      status = session->RunGraph(kGraphId, MakeInputs(kv_layout), outputs);
    }
  } else {
    const auto flow_graph = BuildFlowGraph(
        model_path, graph_config, mode == "serialized-dataflow");
    status = session->AddGraph(kGraphId, flow_graph.ToGeGraph());
    if (status == ge::SUCCESS) status = session->CompileGraph(kGraphId);
    if (status == ge::SUCCESS) {
      ge::DataFlowInfo flow_info;
      status = session->FeedDataFlowGraph(
          kGraphId, MakeInputs(kv_layout), flow_info, kFeedTimeoutMs);
      if (status == ge::SUCCESS) {
        status = session->FetchDataFlowGraph(kGraphId, outputs, flow_info,
                                             kFetchTimeoutMs);
      }
    }
  }
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - started)
                              .count();
  const bool exact = status == ge::SUCCESS && OutputExact(outputs);
  WriteSummary(summary_path, mode, status, exact, elapsed_ms,
               model_load_status, model_valid, jit_compile, kv_layout);
  std::cout << "MINI_FIA_RESULT mode=" << mode
            << " ge_jit_compile=" << jit_compile
            << " status=" << static_cast<uint32_t>(status)
            << " output_exact=" << exact << " elapsed_ms=" << elapsed_ms
            << std::endl;
  session->RemoveGraph(kGraphId);
  session.reset();
  ge::GEFinalize();
  return status == ge::SUCCESS && exact &&
                 model_load_status == ge::GRAPH_SUCCESS && model_valid
             ? 0
             : 10;
}
