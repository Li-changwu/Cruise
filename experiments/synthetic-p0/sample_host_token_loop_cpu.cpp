#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "all_ops.h"
#include "ge/ge_api.h"
#include "graph/graph.h"

using namespace ge;

namespace {
constexpr size_t kStateElements = 4;

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
  ge::Graph graph("HostTokenStateGraph");
  graph.SetInputs({state, delta}).SetOutputs({add});
  return graph;
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

ge::Status RunTokenLoop(const std::shared_ptr<ge::Session> &session,
                        int64_t max_steps, int64_t stop_token,
                        int64_t orchestration_us,
                        std::vector<ge::Tensor> &outputs,
                        int64_t &executed_steps) {
  int32_t initial_state[kStateElements] = {0, 0, 1, 0};
  int32_t delta_data[kStateElements] = {1, 1, 0, 1};
  ge::Tensor state = MakeTensor(initial_state);
  ge::Tensor delta = MakeTensor(delta_data);
  std::vector<ge::Tensor> inputs = {state, delta};
  executed_steps = 0;

  for (int64_t step = 0; step < max_steps; ++step) {
    outputs.clear();
    const auto ret = session->RunGraph(0, inputs, outputs);
    if (ret != ge::SUCCESS || outputs.size() != 1) {
      return ret == ge::SUCCESS ? ge::FAILED : ret;
    }
    ++executed_steps;
    auto *state_data = reinterpret_cast<int32_t *>(outputs[0].GetData());
    if (state_data == nullptr) {
      return ge::FAILED;
    }
    if (orchestration_us > 0) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(orchestration_us));
    }
    if (state_data[0] >= stop_token || executed_steps >= max_steps) {
      state_data[2] = 0;
      break;
    }
    inputs = {outputs[0], delta};
  }
  return ge::SUCCESS;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: sample_host_token_loop MAX_STEPS STOP_TOKEN "
                 "WARMUP_COUNT ORCHESTRATION_US REPS"
              << std::endl;
    return 2;
  }
  const int64_t max_steps = std::stoll(argv[1]);
  const int64_t stop_token = std::stoll(argv[2]);
  const int64_t warmup_count = std::stoll(argv[3]);
  const int64_t orchestration_us = std::stoll(argv[4]);
  const int64_t reps = std::stoll(argv[5]);
  if (max_steps < 1 || max_steps > 1024 || stop_token < 1 ||
      warmup_count < 0 || warmup_count > 10 || orchestration_us < 0 ||
      orchestration_us > 10000 || reps < 1 || reps > 100) {
    std::cerr << "invalid argument" << std::endl;
    return 2;
  }
  const int64_t expected_steps = std::min(max_steps, stop_token);
  std::cout << "HOST_TOKEN_LOOP_START pid=" << getpid()
            << " max_steps=" << max_steps << " stop_token=" << stop_token
            << " orchestration_us=" << orchestration_us << std::endl;

  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "0"}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) {
    std::cerr << "GEInitialize failed ret=" << ret << std::endl;
    return ret;
  }
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, BuildStateGraph());
  if (ret != ge::SUCCESS) {
    std::cerr << "AddGraph failed ret=" << ret << std::endl;
    ge::GEFinalize();
    return ret;
  }

  std::vector<ge::Tensor> outputs;
  int64_t executed_steps = 0;
  for (int64_t warmup = 0; warmup < warmup_count; ++warmup) {
    ret = RunTokenLoop(session, max_steps, stop_token, 0, outputs,
                       executed_steps);
    if (ret != ge::SUCCESS || !CheckState(outputs, expected_steps)) {
      std::cerr << "warmup failed ret=" << ret << std::endl;
      ge::GEFinalize();
      return ret == ge::SUCCESS ? 3 : ret;
    }
  }

  for (int64_t rep = 1; rep <= reps; ++rep) {
    const auto start = std::chrono::steady_clock::now();
    const auto cpu_start_us = ProcessCpuUs();
    ret = RunTokenLoop(session, max_steps, stop_token, orchestration_us,
                       outputs, executed_steps);
    const auto cpu_end_us = ProcessCpuUs();
    const auto end = std::chrono::steady_clock::now();
    if (ret != ge::SUCCESS || executed_steps != expected_steps ||
        !CheckState(outputs, expected_steps)) {
      std::cerr << "run/check failed rep=" << rep << " ret=" << ret
                << " executed=" << executed_steps << std::endl;
      ge::GEFinalize();
      return ret == ge::SUCCESS ? 3 : ret;
    }
    const auto loop_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             end - start)
                             .count();
    std::cout << "HOST_TOKEN_LOOP_RESULT rep=" << rep
              << " executed=" << executed_steps << " loop_us=" << loop_us
              << " host_cpu_us=" << (cpu_end_us - cpu_start_us)
              << " final_state=[" << expected_steps << "," << expected_steps
              << ",0," << expected_steps << "]" << std::endl;
  }
  ge::GEFinalize();
  return 0;
}
