#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"

namespace {
constexpr uint32_t kGraphId = 0;
constexpr int32_t kFeedTimeoutMs = 30000;
constexpr int32_t kFetchTimeoutMs = 300000;
constexpr size_t kResultElements = 19;

int64_t CaseId(const std::string &name) {
  if (name == "tensor-msg-21") return 1;
  if (name == "tensor-msg-28") return 2;
  if (name == "tensor-msg-42") return 3;
  if (name == "factory-wrap-28") return 4;
  if (name == "factory-wrap-42") return 5;
  if (name == "raw-msg-28") return 6;
  if (name == "raw-msg-42") return 7;
  if (name == "tensor-list-2x21") return 8;
  if (name == "tensor-msg-2x42") return 9;
  if (name == "factory-wrap-2x42") return 10;
  if (name == "tensor-msg-4x21") return 11;
  if (name == "tensor-list-4x21") return 12;
  return 0;
}

ge::dflow::FlowGraph BuildFlowGraph(const char *function_config) {
  using namespace ge::dflow;
  auto selector = FlowData("Selector", 0);
  auto function = FunctionPp("p3_allocator_probe_pp")
                      .SetCompileConfig(function_config);
  auto node = FlowNode("p3_allocator_probe_node", 1, 1);
  node.AddPp(function).SetInput(0, selector);
  FlowGraph graph("cruise_p3_allocator_probe");
  graph.SetInputs({selector}).SetOutputs({node});
  return graph;
}

ge::Tensor MakeSelector(int64_t case_id) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape({1}), ge::FORMAT_ND,
                                      ge::DT_INT64));
  tensor.SetData(reinterpret_cast<uint8_t *>(&case_id), sizeof(case_id));
  return tensor;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: p3_allocator_probe_host FUNCTION_CONFIG "
                 "DEPLOY_CONFIG CASE"
              << std::endl;
    return 2;
  }
  const std::string function_config = argv[1];
  const std::string deploy_config = argv[2];
  const std::string case_name = argv[3];
  const int64_t case_id = CaseId(case_name);
  if (access(function_config.c_str(), R_OK) != 0 ||
      access(deploy_config.c_str(), R_OK) != 0 || case_id == 0) {
    std::cerr << "invalid allocator probe config or case" << std::endl;
    return 2;
  }
  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.experiment.data_flow_deploy_info_path", deploy_config.c_str()},
      {"ge.graphRunMode", "0"}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) return ret;
  auto flow_graph = BuildFlowGraph(function_config.c_str());
  const auto &graph = flow_graph.ToGeGraph();
  if (!graph.IsValid()) {
    ge::GEFinalize();
    return 3;
  }
  auto session = std::make_shared<ge::Session>(config);
  ret = session->AddGraph(kGraphId, graph);
  if (ret == ge::SUCCESS) ret = session->CompileGraph(kGraphId);
  if (ret == ge::SUCCESS) {
    ge::DataFlowInfo flow_info;
    ret = session->FeedDataFlowGraph(
        kGraphId, {MakeSelector(case_id)}, flow_info, kFeedTimeoutMs);
  }
  std::vector<ge::Tensor> outputs;
  ge::DataFlowInfo output_info;
  if (ret == ge::SUCCESS) {
    ret = session->FetchDataFlowGraph(
        kGraphId, outputs, output_info, kFetchTimeoutMs);
  }
  std::array<int64_t, kResultElements> values{};
  if (ret != ge::SUCCESS || outputs.size() != 1 ||
      outputs[0].GetData() == nullptr ||
      outputs[0].GetSize() != values.size() * sizeof(values[0])) {
    std::cerr << "P3_ALLOCATOR_PROBE_TRANSPORT_FAILED case=" << case_name
              << " status=" << ret << std::endl;
    session->RemoveGraph(kGraphId);
    session.reset();
    ge::GEFinalize();
    return 4;
  }
  std::memcpy(values.data(), outputs[0].GetData(), outputs[0].GetSize());
  std::cout << "P3_ALLOCATOR_OBSERVATION case=" << case_name
            << " case_id=" << values[0]
            << " api_id=" << values[1]
            << " requested_bytes=" << values[2]
            << " status=" << values[3]
            << " allocated_count=" << values[4]
            << " converted_count=" << values[5]
            << " ret_code=" << values[6]
            << " msg_type=" << values[7]
            << " tensor_count=" << values[8]
            << " shape_ok=" << values[9]
            << " dtype_ok=" << values[10]
            << " elements=" << values[11]
            << " data_bytes=" << values[12]
            << " data_nonnull=" << values[13]
            << " touch_ok=" << values[14]
            << " first=" << values[15]
            << " last=" << values[16]
            << " expected_first=" << values[17]
            << " expected_last=" << values[18] << std::endl;
  const bool valid_observation = values[0] == case_id && values[2] > 0;
  session->RemoveGraph(kGraphId);
  session.reset();
  ge::GEFinalize();
  return valid_observation ? 0 : 5;
}
