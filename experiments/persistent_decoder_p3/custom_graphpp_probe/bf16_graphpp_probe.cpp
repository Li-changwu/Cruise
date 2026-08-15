#include <algorithm>
#include <chrono>
#include <cstdint>
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
constexpr size_t kElements = 18944U;
constexpr size_t kBytes = kElements * sizeof(uint16_t);

bool ReadInput(const std::string &path, std::vector<uint8_t> &data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || static_cast<size_t>(stream.tellg()) != kBytes) return false;
  data.resize(kBytes);
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(data.data()),
              static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(stream);
}

ge::Tensor MakeInput(std::vector<uint8_t> &data) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape({1, 1, 18944}),
                                      ge::FORMAT_ND, ge::DT_BF16));
  tensor.SetData(data.data(), data.size());
  return tensor;
}

ge::Graph LoadAir(const std::string &air_path, uint32_t *load_status = nullptr,
                  bool *valid = nullptr) {
  ge::Graph graph("CruiseBf16MaterializeProbe");
  const auto status = graph.LoadFromFile(air_path.c_str());
  if (load_status != nullptr) *load_status = static_cast<uint32_t>(status);
  if (valid != nullptr) *valid = graph.IsValid();
  std::cout << "BF16_GRAPHPP_AIR_LOAD status=" << status
            << " valid=" << graph.IsValid() << std::endl;
  return graph;
}

ge::dflow::FlowGraph BuildFlowGraph(const std::string &air_path,
                                    const std::string &graph_config) {
  using namespace ge::dflow;
  auto input = FlowData("x", 0);
  auto materialize =
      GraphPp("bf16_materialize_graph_pp",
              [air_path]() { return LoadAir(air_path); })
          .SetCompileConfig(graph_config.c_str());
  auto node = FlowNode("bf16_materialize_node", 1, 1);
  node.AddPp(materialize).SetInput(0, input);
  FlowGraph graph("cruise_bf16_materialize_dataflow");
  graph.SetInputs({input}).SetOutputs({node});
  return graph;
}

bool OutputExact(const std::vector<uint8_t> &input,
                 const std::vector<ge::Tensor> &outputs) {
  if (outputs.size() != 1 || outputs[0].GetData() == nullptr ||
      outputs[0].GetSize() != input.size()) {
    return false;
  }
  const auto *actual =
      reinterpret_cast<const uint8_t *>(outputs[0].GetData());
  return std::equal(input.begin(), input.end(), actual);
}

bool WriteOutput(const std::string &path,
                 const std::vector<ge::Tensor> &outputs) {
  if (outputs.size() != 1 || outputs[0].GetData() == nullptr ||
      outputs[0].GetSize() != kBytes) {
    return false;
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return false;
  stream.write(reinterpret_cast<const char *>(outputs[0].GetData()),
               static_cast<std::streamsize>(outputs[0].GetSize()));
  return static_cast<bool>(stream);
}

void WriteSummary(const std::string &path, const std::string &mode,
                  ge::Status status, ge::Status add_status,
                  ge::Status compile_status, ge::Status feed_status,
                  ge::Status fetch_status, uint32_t model_load_status,
                  bool model_valid, bool exact, bool output_saved,
                  int64_t elapsed_ms) {
  std::ofstream stream(path, std::ios::trunc);
  const bool pass = status == ge::SUCCESS &&
                    model_load_status == ge::GRAPH_SUCCESS && model_valid &&
                    exact && output_saved;
  stream << "{\n"
         << "  \"gate\": \"P5-CUSTOM-AICORE-GRAPHPP\",\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"mode\": \"" << mode << "\",\n"
         << "  \"status\": " << static_cast<uint32_t>(status) << ",\n"
         << "  \"add_status\": " << static_cast<uint32_t>(add_status)
         << ",\n"
         << "  \"compile_status\": "
         << static_cast<uint32_t>(compile_status) << ",\n"
         << "  \"feed_status\": " << static_cast<uint32_t>(feed_status)
         << ",\n"
         << "  \"fetch_status\": " << static_cast<uint32_t>(fetch_status)
         << ",\n"
         << "  \"model_load_status\": " << model_load_status << ",\n"
         << "  \"model_valid\": " << (model_valid ? "true" : "false")
         << ",\n"
         << "  \"output_exact\": " << (exact ? "true" : "false")
         << ",\n"
         << "  \"output_saved\": " << (output_saved ? "true" : "false")
         << ",\n"
         << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
         << "  \"claim_boundary\": \"Single BF16 identity custom AICore op only; no FIA, Owner, or P5 performance claim.\"\n"
         << "}\n";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: bf16_graphpp_probe MODE AIR GRAPH_CONFIG "
                 "DEPLOY_CONFIG INPUT OUTPUT SUMMARY_JSON"
              << std::endl;
    return 2;
  }
  const std::string mode = argv[1];
  const std::string air_path = argv[2];
  const std::string graph_config = argv[3];
  const std::string deploy_config = argv[4];
  const std::string input_path = argv[5];
  const std::string output_path = argv[6];
  const std::string summary_path = argv[7];
  if ((mode != "graph" && mode != "dataflow") ||
      access(air_path.c_str(), R_OK) != 0 ||
      access(graph_config.c_str(), R_OK) != 0 ||
      access(deploy_config.c_str(), R_OK) != 0 ||
      access(input_path.c_str(), R_OK) != 0) {
    return 2;
  }

  std::vector<uint8_t> input;
  if (!ReadInput(input_path, input)) return 3;
  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
  };
  if (mode == "dataflow") {
    config["ge.exec.logicalDeviceClusterDeployMode"] = "SINGLE";
    config["ge.exec.logicalDeviceId"] = "[0:0]";
    config["ge.experiment.data_flow_deploy_info_path"] =
        deploy_config.c_str();
  }

  auto status = ge::GEInitialize(config);
  ge::Status add_status = ge::FAILED;
  ge::Status compile_status = ge::FAILED;
  ge::Status feed_status = ge::FAILED;
  ge::Status fetch_status = ge::FAILED;
  uint32_t model_load_status = UINT32_MAX;
  bool model_valid = false;
  if (status != ge::SUCCESS) {
    WriteSummary(summary_path, mode, status, add_status, compile_status,
                 feed_status, fetch_status, model_load_status, model_valid,
                 false, false, 0);
    return 10;
  }

  auto session = std::make_shared<ge::Session>(config);
  const auto started = std::chrono::steady_clock::now();
  auto source_graph = LoadAir(air_path, &model_load_status, &model_valid);
  std::vector<ge::Tensor> inputs = {MakeInput(input)};
  std::vector<ge::Tensor> outputs;
  if (mode == "graph") {
    add_status = session->AddGraph(kGraphId, source_graph);
    status = add_status;
    if (status == ge::SUCCESS) {
      status = session->RunGraph(kGraphId, inputs, outputs);
    }
  } else {
    auto flow_graph = BuildFlowGraph(air_path, graph_config);
    add_status = session->AddGraph(kGraphId, flow_graph.ToGeGraph());
    status = add_status;
    if (status == ge::SUCCESS) {
      compile_status = session->CompileGraph(kGraphId);
      status = compile_status;
    }
    if (status == ge::SUCCESS) {
      ge::DataFlowInfo flow_info;
      feed_status = session->FeedDataFlowGraph(
          kGraphId, inputs, flow_info, kFeedTimeoutMs);
      status = feed_status;
      if (status == ge::SUCCESS) {
        fetch_status = session->FetchDataFlowGraph(
            kGraphId, outputs, flow_info, kFetchTimeoutMs);
        status = fetch_status;
      }
    }
  }
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - started)
                              .count();
  const bool exact = status == ge::SUCCESS && OutputExact(input, outputs);
  const bool output_saved = exact && WriteOutput(output_path, outputs);
  WriteSummary(summary_path, mode, status, add_status, compile_status,
               feed_status, fetch_status, model_load_status, model_valid,
               exact, output_saved, elapsed_ms);
  std::cout << "BF16_GRAPHPP_RESULT mode=" << mode
            << " status=" << static_cast<uint32_t>(status)
            << " output_exact=" << exact
            << " output_saved=" << output_saved
            << " elapsed_ms=" << elapsed_ms << std::endl;
  if (add_status == ge::SUCCESS) session->RemoveGraph(kGraphId);
  session.reset();
  ge::GEFinalize();
  return status == ge::SUCCESS && model_load_status == ge::GRAPH_SUCCESS &&
                 model_valid && exact && output_saved
             ? 0
             : 10;
}
