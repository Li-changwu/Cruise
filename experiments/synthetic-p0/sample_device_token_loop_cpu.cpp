#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include "all_ops.h"
#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "graph/graph.h"

using namespace ge;
using namespace dflow;

namespace {
constexpr size_t kStateElements = 4;
constexpr int32_t kFeedTimeoutMs = 3000;
constexpr int32_t kFetchTimeoutMs = 300000;

int64_t ProcessCpuUs() {
  timespec value{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) {
    return -1;
  }
  return static_cast<int64_t>(value.tv_sec) * 1000000 +
         static_cast<int64_t>(value.tv_nsec) / 1000;
}

ge::Graph BuildStateGraph() {
  auto state = op::Data("state").set_attr_index(0);
  auto delta = op::Data("delta").set_attr_index(1);
  auto add = op::Add("state_update").set_input_x1(state).set_input_x2(delta);
  ge::Graph graph("DeviceTokenStateGraph");
  graph.SetInputs({state, delta}).SetOutputs({add});
  return graph;
}

dflow::FlowGraph BuildFlowGraph(int64_t max_steps, int64_t stop_token) {
  auto state = FlowData("State", 0);
  auto delta = FlowData("Delta", 1);
  auto closure = GraphPp("invoke_state_graph", BuildStateGraph)
                     .SetCompileConfig("../config/add_state_graph.json");
  auto controller = FunctionPp("token_controller_pp")
                        .SetCompileConfig("../config/controller_token_func.json")
                        .SetInitParam("max_steps", max_steps)
                        .SetInitParam("stop_token", stop_token);
  controller.AddInvokedClosure("invoke_graph", closure);
  auto node = FlowNode("token_controller", 2, 1);
  node.AddPp(controller);
  node.SetInput(0, state).SetInput(1, delta);
  dflow::FlowGraph flow_graph("device_token_control_flow");
  flow_graph.SetInputs({state, delta}).SetOutputs({node});
  return flow_graph;
}

ge::Tensor MakeTensor(int32_t *data) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(
      ge::TensorDesc(ge::Shape({static_cast<int64_t>(kStateElements)}),
                     ge::FORMAT_ND, ge::DT_INT32));
  tensor.SetData(reinterpret_cast<uint8_t *>(data),
                 kStateElements * sizeof(int32_t));
  return tensor;
}

bool CheckState(const std::vector<ge::Tensor> &outputs, int64_t expected_steps) {
  if (outputs.size() != 1 ||
      outputs[0].GetSize() != kStateElements * sizeof(int32_t)) {
    return false;
  }
  const auto *state = reinterpret_cast<const int32_t *>(outputs[0].GetData());
  if (state == nullptr) {
    return false;
  }
  const bool correct = state[0] == expected_steps &&
                       state[1] == expected_steps && state[2] == 0 &&
                       state[3] == expected_steps;
  if (!correct) {
    std::cerr << "state mismatch expected_steps=" << expected_steps
              << " actual=[" << state[0] << "," << state[1] << ","
              << state[2] << "," << state[3] << "]" << std::endl;
  }
  return correct;
}

ge::Status RunTransaction(const std::shared_ptr<ge::Session> &session,
                          int64_t expected_steps, int64_t &feed_us,
                          int64_t &fetch_us) {
  int32_t initial_state[kStateElements] = {0, 0, 1, 0};
  int32_t delta_data[kStateElements] = {1, 1, 0, 1};
  std::vector<ge::Tensor> inputs = {MakeTensor(initial_state),
                                    MakeTensor(delta_data)};
  ge::DataFlowInfo flow_info;
  const auto feed_start = std::chrono::steady_clock::now();
  auto ret = session->FeedDataFlowGraph(0, inputs, flow_info, kFeedTimeoutMs);
  const auto feed_end = std::chrono::steady_clock::now();
  if (ret != ge::SUCCESS) {
    return ret;
  }
  std::vector<ge::Tensor> outputs;
  ret = session->FetchDataFlowGraph(0, outputs, flow_info, kFetchTimeoutMs);
  const auto fetch_end = std::chrono::steady_clock::now();
  if (ret != ge::SUCCESS || !CheckState(outputs, expected_steps)) {
    return ret == ge::SUCCESS ? ge::FAILED : ret;
  }
  feed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                feed_end - feed_start)
                .count();
  fetch_us = std::chrono::duration_cast<std::chrono::microseconds>(
                 fetch_end - feed_end)
                 .count();
  return ge::SUCCESS;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: sample_device_token_loop MAX_STEPS STOP_TOKEN "
                 "WARMUP_COUNT REPS"
              << std::endl;
    return 2;
  }
  const int64_t max_steps = std::stoll(argv[1]);
  const int64_t stop_token = std::stoll(argv[2]);
  const int64_t warmup_count = std::stoll(argv[3]);
  const int64_t reps = std::stoll(argv[4]);
  if (max_steps < 1 || max_steps > 1024 || stop_token < 1 ||
      warmup_count < 0 || warmup_count > 10 || reps < 1 || reps > 100) {
    std::cerr << "invalid argument" << std::endl;
    return 2;
  }
  const int64_t expected_steps = std::min(max_steps, stop_token);
  std::cout << "DEVICE_TOKEN_LOOP_START pid=" << getpid()
            << " max_steps=" << max_steps << " stop_token=" << stop_token
            << std::endl;

  auto flow_graph = BuildFlowGraph(max_steps, stop_token);
  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.graphRunMode", "0"}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) {
    std::cerr << "GEInitialize failed ret=" << ret << std::endl;
    return ret;
  }
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, flow_graph.ToGeGraph());
  if (ret != ge::SUCCESS) {
    std::cerr << "AddGraph failed ret=" << ret << std::endl;
    ge::GEFinalize();
    return ret;
  }

  int64_t feed_us = 0;
  int64_t fetch_us = 0;
  for (int64_t warmup = 0; warmup < warmup_count; ++warmup) {
    ret = RunTransaction(session, expected_steps, feed_us, fetch_us);
    if (ret != ge::SUCCESS) {
      std::cerr << "warmup failed ret=" << ret << std::endl;
      ge::GEFinalize();
      return ret;
    }
  }
  for (int64_t rep = 1; rep <= reps; ++rep) {
    const auto cpu_start_us = ProcessCpuUs();
    ret = RunTransaction(session, expected_steps, feed_us, fetch_us);
    const auto cpu_end_us = ProcessCpuUs();
    if (ret != ge::SUCCESS) {
      std::cerr << "run/check failed rep=" << rep << " ret=" << ret
                << std::endl;
      ge::GEFinalize();
      return ret;
    }
    std::cout << "DEVICE_TOKEN_LOOP_RESULT rep=" << rep
              << " executed=" << expected_steps << " feed_us=" << feed_us
              << " fetch_us=" << fetch_us
              << " host_cpu_us=" << (cpu_end_us - cpu_start_us)
              << " feed_calls=1 fetch_calls=1 final_state=["
              << expected_steps << "," << expected_steps << ",0,"
              << expected_steps << "]" << std::endl;
  }
  ge::GEFinalize();
  return 0;
}
