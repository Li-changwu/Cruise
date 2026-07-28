#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "all_ops.h"
#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"

namespace {
constexpr size_t kElements = 16;
constexpr int32_t kFeedTimeoutMs = 60000;
constexpr int32_t kFetchTimeoutMs = 300000;

ge::dflow::FlowGraph BuildDataFlow(const std::string &func_config) {
  auto input = ge::dflow::FlowData("bf16_input", 0);
  auto pp = ge::dflow::FunctionPp("bf16_echo_pp")
                .SetCompileConfig(func_config.c_str());
  auto node = ge::dflow::FlowNode("bf16_echo_node", 1, 1);
  node.AddPp(pp).SetInput(0, input);
  ge::dflow::FlowGraph graph("attempt66a_r6_r2_bf16_udf_echo");
  graph.SetInputs({input}).SetOutputs({node});
  return graph;
}

ge::Tensor MakeInput(std::array<uint16_t, kElements> &bits) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(
      ge::Shape({static_cast<int64_t>(kElements)}), ge::FORMAT_ND,
      ge::DT_BF16));
  tensor.SetData(reinterpret_cast<uint8_t *>(bits.data()),
                 bits.size() * sizeof(uint16_t));
  return tensor;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: bf16_udf_echo_smoke FUNCTION_CONFIG\n";
    return 2;
  }
  const auto flow_graph = BuildDataFlow(argv[1]);
  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.graphRunMode", "0"}};
  auto ret = ge::GEInitialize(options);
  if (ret != ge::SUCCESS) {
    std::cerr << "GE_INITIALIZE_FAILED ret=" << ret << std::endl;
    return 3;
  }
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, flow_graph.ToGeGraph());
  if (ret != ge::SUCCESS) {
    std::cerr << "ADD_GRAPH_FAILED ret=" << ret << std::endl;
    session.reset();
    ge::GEFinalize();
    return 4;
  }

  std::array<uint16_t, kElements> input_bits;
  for (size_t index = 0; index < input_bits.size(); ++index) {
    input_bits[index] = static_cast<uint16_t>(0x3f80U + index);
  }
  std::vector<ge::Tensor> inputs = {MakeInput(input_bits)};
  std::cout << "HOST_INPUT dtype="
            << static_cast<int>(inputs[0].GetTensorDesc().GetDataType())
            << " bytes=" << inputs[0].GetSize() << std::endl;
  ge::DataFlowInfo flow_info;
  ret = session->FeedDataFlowGraph(0, inputs, flow_info, kFeedTimeoutMs);
  if (ret != ge::SUCCESS) {
    std::cerr << "FEED_FAILED ret=" << ret << std::endl;
    session.reset();
    ge::GEFinalize();
    return 5;
  }

  std::vector<ge::Tensor> outputs;
  ret = session->FetchDataFlowGraph(0, outputs, flow_info, kFetchTimeoutMs);
  if (ret != ge::SUCCESS) {
    std::cerr << "FETCH_FAILED ret=" << ret << std::endl;
    session.reset();
    ge::GEFinalize();
    return 6;
  }
  const size_t expected_bytes = input_bits.size() * sizeof(uint16_t);
  if (outputs.size() != 1 || outputs[0].GetData() == nullptr ||
      outputs[0].GetSize() != expected_bytes ||
      outputs[0].GetTensorDesc().GetDataType() != ge::DT_BF16 ||
      std::memcmp(outputs[0].GetData(), input_bits.data(), expected_bytes) != 0) {
    std::cerr << "OUTPUT_MISMATCH count=" << outputs.size();
    if (!outputs.empty()) {
      std::cerr << " dtype="
                << static_cast<int>(
                       outputs[0].GetTensorDesc().GetDataType())
                << " bytes=" << outputs[0].GetSize();
    }
    std::cerr << std::endl;
    session.reset();
    ge::GEFinalize();
    return 7;
  }
  session.reset();
  ge::GEFinalize();
  std::cout << "ATTEMPT66A_R6_R2_BF16_UDF_ECHO_PASS feed_calls=1 fetch_calls=1"
            << std::endl;
  return 0;
}
