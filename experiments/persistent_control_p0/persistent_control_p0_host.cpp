#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"

namespace {
constexpr uint32_t kGraphId = 0;
constexpr int32_t kFeedTimeoutMs = 30000;
constexpr int32_t kFetchTimeoutMs = 300000;
constexpr int32_t kFetchStartupRetryDelayMs = 100;
constexpr int32_t kFetchStartupRetryLimit = 1200;
constexpr int64_t kEventAdmit = 1;
constexpr int64_t kEventCancel = 2;
constexpr int64_t kEventShutdown = 3;
constexpr int64_t kOutputCommit = 2;
constexpr int64_t kOutputRetireCancelled = 4;
constexpr int64_t kOutputQuiescent = 5;
constexpr int64_t kOutputShutdown = 6;
constexpr int64_t kTargetQuanta = 1024;
constexpr int64_t kCancelAfterCommits = 128;

struct SharedState {
  std::mutex mutex;
  std::condition_variable changed;
  std::map<int64_t, int64_t> commits;
  int64_t quiescent = 0;
  int64_t fetch_calls = 0;
  int64_t rejected = 0;
  bool cancelled_generation_three = false;
  bool shutdown = false;
  bool stop = false;
  int32_t fetch_status = ge::SUCCESS;
};

int64_t ProcessCpuUs() {
  timespec value{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) {
    return -1;
  }
  return static_cast<int64_t>(value.tv_sec) * 1000000 +
         static_cast<int64_t>(value.tv_nsec) / 1000;
}

ge::dflow::FlowGraph BuildFlowGraph(const char *function_config,
                                    int64_t owner_instance_id,
                                    int64_t quantum_delay_us) {
  using namespace ge::dflow;
  auto event = FlowData("Event", 0);
  auto controller = FunctionPp("persistent_control_p0_pp")
                        .SetCompileConfig(function_config)
                        .SetInitParam("owner_instance_id", owner_instance_id)
                        .SetInitParam("quantum_delay_us", quantum_delay_us);
  auto node = FlowNode("persistent_control_p0_node", 1, 1);
  node.AddPp(controller);
  node.SetInput(0, event);
  FlowGraph graph("cruise_persistent_control_p0");
  std::vector<FlowOperator> inputs = {event};
  std::vector<std::pair<FlowOperator, std::vector<size_t>>> outputs = {
      {node, {0}}};
  graph.SetInputs(inputs).SetOutputs(outputs).SetContainsNMappingNode(true);
  return graph;
}

ge::Tensor MakeEvent(int64_t owner, int64_t type, int64_t event_seq,
                     int64_t generation, int64_t target_quanta,
                     int64_t cancel_seq, int64_t seed) {
  std::array<int64_t, 8> values = {owner, type, event_seq, generation,
                                   target_quanta, cancel_seq, seed, 0};
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape({8}), ge::FORMAT_ND,
                                      ge::DT_INT64));
  tensor.SetData(reinterpret_cast<uint8_t *>(values.data()),
                 values.size() * sizeof(values[0]));
  return tensor;
}

bool ParseOutput(const std::vector<ge::Tensor> &outputs,
                 std::array<int64_t, 8> &values) {
  if (outputs.size() != 1 || outputs[0].GetSize() != values.size() * sizeof(int64_t) ||
      outputs[0].GetData() == nullptr) {
    return false;
  }
  std::memcpy(values.data(), outputs[0].GetData(),
              values.size() * sizeof(values[0]));
  return true;
}

void FetchOutputs(const std::shared_ptr<ge::Session> &session,
                  SharedState &shared) {
  int32_t startup_failures = 0;
  bool fetched_output = false;
  while (true) {
    std::vector<ge::Tensor> outputs;
    ge::DataFlowInfo flow_info;
    const auto ret = session->FetchDataFlowGraph(
        kGraphId, outputs, flow_info, kFetchTimeoutMs);
    if (ret != ge::SUCCESS) {
      if (!fetched_output && startup_failures < kFetchStartupRetryLimit) {
        {
          std::lock_guard<std::mutex> lock(shared.mutex);
          if (shared.stop) {
            return;
          }
        }
        ++startup_failures;
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kFetchStartupRetryDelayMs));
        continue;
      }
      std::lock_guard<std::mutex> lock(shared.mutex);
      shared.fetch_status = ret;
      shared.changed.notify_all();
      return;
    }
    fetched_output = true;
    std::array<int64_t, 8> values{};
    if (!ParseOutput(outputs, values)) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      shared.fetch_status = ge::FAILED;
      shared.changed.notify_all();
      return;
    }
    std::cout << "P0_OUTPUT owner=" << values[0]
              << " type=" << values[1]
              << " event_seq=" << values[2]
              << " generation=" << values[3]
              << " commit_seq=" << values[4]
              << " state=" << values[5]
              << " remaining=" << values[6]
              << " status=" << values[7]
              << " transaction=" << flow_info.GetTransactionId()
              << " flags=" << flow_info.GetFlowFlags() << std::endl;
    {
      std::lock_guard<std::mutex> lock(shared.mutex);
      ++shared.fetch_calls;
      if (values[1] == kOutputCommit) {
        shared.commits[values[3]] = values[4];
      } else if (values[1] == kOutputRetireCancelled && values[3] == 3) {
        shared.cancelled_generation_three = true;
      } else if (values[1] == kOutputQuiescent) {
        ++shared.quiescent;
      } else if (values[1] == kOutputShutdown) {
        shared.shutdown = true;
      } else if (values[1] == 7) {
        ++shared.rejected;
      }
      shared.changed.notify_all();
      if (shared.shutdown) {
        return;
      }
    }
  }
}

template <typename Predicate>
bool WaitFor(SharedState &shared, Predicate predicate, int seconds) {
  std::unique_lock<std::mutex> lock(shared.mutex);
  return shared.changed.wait_for(lock, std::chrono::seconds(seconds), [&]() {
    return predicate(shared) || shared.fetch_status != ge::SUCCESS;
  }) && predicate(shared) && shared.fetch_status == ge::SUCCESS;
}

ge::Status FeedEvent(const std::shared_ptr<ge::Session> &session, int64_t owner,
                     int64_t type, int64_t event_seq, int64_t generation,
                     int64_t target_quanta, int64_t cancel_seq, int64_t seed) {
  std::vector<ge::Tensor> inputs = {MakeEvent(
      owner, type, event_seq, generation, target_quanta, cancel_seq, seed)};
  ge::DataFlowInfo flow_info;
  flow_info.SetTransactionId(static_cast<uint64_t>(event_seq));
  return session->FeedDataFlowGraph(kGraphId, inputs, flow_info,
                                    kFeedTimeoutMs);
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "usage: persistent_control_p0_host FUNCTION_CONFIG "
                 "DEPLOY_CONFIG OWNER_INSTANCE_ID QUANTUM_DELAY_US "
                 "[placement_probe]"
              << std::endl;
    return 2;
  }
  const std::string function_config = argv[1];
  const std::string deploy_config = argv[2];
  const int64_t owner = std::stoll(argv[3]);
  const int64_t quantum_delay_us = std::stoll(argv[4]);
  const bool placement_probe = argc == 6 && std::string(argv[5]) == "placement_probe";
  if (access(function_config.c_str(), R_OK) != 0 ||
      access(deploy_config.c_str(), R_OK) != 0 || owner <= 0 ||
      quantum_delay_us < 0 || quantum_delay_us > 1000000 ||
      (argc == 6 && !placement_probe)) {
    std::cerr << "invalid configuration path, owner, or quantum delay"
              << std::endl;
    return 2;
  }

  std::cout << "P0_OWNER_START pid=" << getpid() << " owner=" << owner
            << " quantum_delay_us=" << quantum_delay_us << std::endl;
  const int64_t cpu_start_us = ProcessCpuUs();
  const auto wall_start = std::chrono::steady_clock::now();
  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.experiment.data_flow_deploy_info_path", deploy_config.c_str()},
      {"ge.graphRunMode", "0"}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) {
    std::cerr << "GEInitialize failed ret=" << ret << std::endl;
    return ret;
  }
  auto flow_graph = BuildFlowGraph(function_config.c_str(), owner,
                                   quantum_delay_us);
  const auto &graph = flow_graph.ToGeGraph();
  if (!graph.IsValid()) {
    std::cerr << "BuildFlowGraph produced an invalid graph" << std::endl;
    ge::GEFinalize();
    return 1;
  }
  auto session = std::make_shared<ge::Session>(config);
  ret = session->AddGraph(kGraphId, graph);
  if (ret != ge::SUCCESS) {
    std::cerr << "AddGraph failed ret=" << ret << std::endl;
    session.reset();
    ge::GEFinalize();
    return ret;
  }
  ret = session->CompileGraph(kGraphId);
  if (ret != ge::SUCCESS) {
    std::cerr << "CompileGraph failed ret=" << ret
              << " error=" << ge::GEGetErrorMsg() << std::endl;
    session->RemoveGraph(kGraphId);
    session.reset();
    ge::GEFinalize();
    return ret;
  }

  SharedState shared;
  int64_t feed_calls = 0;
  auto send = [&](int64_t type, int64_t event_seq, int64_t generation,
                  int64_t target_quanta, int64_t cancel_seq,
                  int64_t seed) -> bool {
    const auto status = FeedEvent(session, owner, type, event_seq, generation,
                                  target_quanta, cancel_seq, seed);
    if (status != ge::SUCCESS) {
      std::cerr << "FeedDataFlowGraph failed event_seq=" << event_seq
                << " ret=" << status << std::endl;
      {
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.stop = true;
        shared.changed.notify_all();
      }
      return false;
    }
    ++feed_calls;
    return true;
  };

  if (placement_probe) {
    std::thread drain(FetchOutputs, session, std::ref(shared));
    const bool passed =
        send(kEventAdmit, 1, 1, 1, 0, 1000) &&
        WaitFor(shared, [](const SharedState &s) {
          return s.quiescent >= 1 && s.commits.count(1) != 0 &&
                 s.commits.at(1) == 1;
        }, 120) &&
        send(kEventShutdown, 2, 0, 0, 0, 0) &&
        WaitFor(shared, [](const SharedState &s) { return s.shutdown; }, 120);
    if (drain.joinable()) {
      drain.join();
    }
    const auto remove_status = session->RemoveGraph(kGraphId);
    session.reset();
    const auto finalize_status = ge::GEFinalize();
    int64_t commits = 0;
    int64_t quiescent = 0;
    int32_t fetch_status = ge::FAILED;
    {
      std::lock_guard<std::mutex> lock(shared.mutex);
      commits = shared.commits[1];
      quiescent = shared.quiescent;
      fetch_status = shared.fetch_status;
    }
    std::cout << "P0_PLACEMENT_PROBE owner=" << owner
              << " feed_calls=" << feed_calls << " commits=" << commits
              << " quiescent=" << quiescent
              << " compile_status=" << ge::SUCCESS
              << " fetch_status=" << fetch_status
              << " remove_status=" << remove_status
              << " finalize_status=" << finalize_status << std::endl;
    return passed && feed_calls == 2 && commits == 1 && quiescent == 1 &&
                   fetch_status == ge::SUCCESS && remove_status == ge::SUCCESS &&
                   finalize_status == ge::SUCCESS
               ? 0
               : 1;
  }

  std::thread drain(FetchOutputs, session, std::ref(shared));
  bool passed = send(kEventAdmit, 1, 1, kTargetQuanta, 0, 1000) &&
                WaitFor(shared, [](const SharedState &s) {
                  return s.quiescent >= 1 && s.commits.count(1) != 0 &&
                         s.commits.at(1) == kTargetQuanta;
                }, 120) &&
                send(kEventAdmit, 2, 2, kTargetQuanta, 0, 2000) &&
                WaitFor(shared, [](const SharedState &s) {
                  return s.quiescent >= 2 && s.commits.count(2) != 0 &&
                         s.commits.at(2) == kTargetQuanta;
                }, 120) &&
                send(kEventAdmit, 3, 3, kTargetQuanta, 0, 3000) &&
                WaitFor(shared, [](const SharedState &s) {
                  return s.commits.count(3) != 0 &&
                         s.commits.at(3) >= kCancelAfterCommits;
                }, 120) &&
                send(kEventCancel, 4, 3, 0, 1, 0) &&
                WaitFor(shared, [](const SharedState &s) {
                  return s.quiescent >= 3 && s.cancelled_generation_three;
                }, 120) &&
                send(kEventAdmit, 5, 4, kTargetQuanta, 0, 4000) &&
                WaitFor(shared, [](const SharedState &s) {
                  return s.quiescent >= 4 && s.commits.count(4) != 0 &&
                         s.commits.at(4) == kTargetQuanta;
                }, 120) &&
                send(kEventShutdown, 6, 0, 0, 0, 0) &&
                WaitFor(shared, [](const SharedState &s) {
                  return s.shutdown;
                }, 120);

  if (!passed) {
    std::cerr << "P0 sequence failed or timed out" << std::endl;
  }
  if (drain.joinable()) {
    drain.join();
  }
  const auto remove_status = session->RemoveGraph(kGraphId);
  session.reset();
  const auto finalize_status = ge::GEFinalize();
  const int64_t cpu_end_us = ProcessCpuUs();
  const auto wall_end = std::chrono::steady_clock::now();

  int64_t commits_one = 0;
  int64_t commits_two = 0;
  int64_t commits_three = 0;
  int64_t commits_four = 0;
  int64_t fetch_calls = 0;
  int64_t quiescent = 0;
  int64_t rejected = 0;
  int32_t fetch_status = ge::FAILED;
  {
    std::lock_guard<std::mutex> lock(shared.mutex);
    commits_one = shared.commits[1];
    commits_two = shared.commits[2];
    commits_three = shared.commits[3];
    commits_four = shared.commits[4];
    fetch_calls = shared.fetch_calls;
    quiescent = shared.quiescent;
    rejected = shared.rejected;
    fetch_status = shared.fetch_status;
  }
  const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           wall_end - wall_start)
                           .count();
  std::cout << "P0_SUMMARY owner=" << owner
            << " owner_starts=1 feed_calls=" << feed_calls
            << " fetch_calls=" << fetch_calls
            << " admission_events=4 cancel_events=1 shutdown_events=1"
            << " quiescent=" << quiescent
            << " rejected=" << rejected
            << " generation1_commits=" << commits_one
            << " generation2_commits=" << commits_two
            << " generation3_commits=" << commits_three
            << " generation4_commits=" << commits_four
            << " host_cpu_us=" << (cpu_end_us - cpu_start_us)
            << " wall_ms=" << wall_ms
            << " aicore_tasks_expected=0"
            << " compile_status=" << ge::SUCCESS
            << " fetch_status=" << fetch_status
            << " remove_status=" << remove_status
            << " finalize_status=" << finalize_status << std::endl;
  if (!passed || feed_calls != 6 || rejected != 0 ||
      commits_one != kTargetQuanta || commits_two != kTargetQuanta ||
      commits_three < kCancelAfterCommits || commits_three >= kTargetQuanta ||
      commits_four != kTargetQuanta || quiescent != 4 ||
      fetch_status != ge::SUCCESS || remove_status != ge::SUCCESS ||
      finalize_status != ge::SUCCESS) {
    return 1;
  }
  return 0;
}
