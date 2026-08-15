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
#include <utility>
#include <vector>

#include <unistd.h>

#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "graph/graph.h"

namespace {
constexpr uint32_t kGraphId = 0;
constexpr size_t kRows = 4;
constexpr size_t kEventElements = 16;
constexpr size_t kOutputElements = 12;
constexpr int32_t kFeedTimeoutMs = 30000;
constexpr int32_t kFetchTimeoutMs = 300000;
constexpr int32_t kFetchStartupRetryDelayMs = 100;
constexpr int32_t kFetchStartupRetryLimit = 1200;
constexpr int64_t kEventAdmit = 1;
constexpr int64_t kEventCredit = 2;
constexpr int64_t kEventShutdown = 3;
constexpr int64_t kOutputCommit = 2;
constexpr int64_t kOutputRetireComplete = 3;
constexpr int64_t kOutputQuiescent = 4;
constexpr int64_t kOutputShutdown = 6;
constexpr int64_t kOutputRejected = 7;
constexpr int64_t kTargetSteps = 256;
constexpr int64_t kExpectedAicoreCalls = 752;
constexpr int64_t kExpectedCommits = 1280;

using RowKey = std::pair<int64_t, int64_t>;

struct SharedState {
  std::mutex mutex;
  std::condition_variable changed;
  std::map<RowKey, int64_t> commits;
  std::map<RowKey, bool> retired;
  int64_t fetch_calls = 0;
  int64_t quiescent = 0;
  int64_t rejected = 0;
  int64_t aicore_calls = 0;
  int64_t total_commits = 0;
  bool shutdown = false;
  bool stop = false;
  int32_t fetch_status = ge::SUCCESS;
};

bool CommitIs(const SharedState &state, int64_t cohort, int64_t row,
              int64_t expected) {
  const auto found = state.commits.find({cohort, row});
  return found != state.commits.end() && found->second == expected;
}

bool IsRetired(const SharedState &state, int64_t cohort, int64_t row) {
  const auto found = state.retired.find({cohort, row});
  return found != state.retired.end() && found->second;
}

int64_t ProcessCpuUs() {
  timespec value{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) return -1;
  return static_cast<int64_t>(value.tv_sec) * 1000000LL +
         static_cast<int64_t>(value.tv_nsec) / 1000LL;
}

ge::Graph BuildRecurrenceGraph() {
  auto state = ge::op::Data("state").set_attr_index(0);
  auto delta = ge::op::Data("delta").set_attr_index(1);
  auto add = ge::op::Add("persistent_recurrence_p1_add")
                 .set_input_x1(state)
                 .set_input_x2(delta);
  ge::Graph graph("PersistentRecurrenceP1AddGraph");
  graph.SetInputs({state, delta}).SetOutputs({add});
  return graph;
}

ge::dflow::FlowGraph BuildFlowGraph(const char *function_config,
                                    const char *graph_config,
                                    int64_t owner) {
  using namespace ge::dflow;
  auto event = FlowData("Event", 0);
  auto recurrence = GraphPp("persistent_recurrence_p1_graph_pp",
                            BuildRecurrenceGraph)
                        .SetCompileConfig(graph_config);
  auto controller = FunctionPp("persistent_recurrence_p1_pp")
                        .SetCompileConfig(function_config)
                        .SetInitParam("owner_instance_id", owner);
  controller.AddInvokedClosure("recurrence_graph", recurrence);
  auto node = FlowNode("persistent_recurrence_p1_node", 1, 1);
  node.AddPp(controller).SetInput(0, event);
  FlowGraph graph("cruise_persistent_recurrence_p1");
  std::vector<FlowOperator> inputs = {event};
  std::vector<std::pair<FlowOperator, std::vector<size_t>>> outputs = {
      {node, {0}}};
  graph.SetInputs(inputs).SetOutputs(outputs).SetContainsNMappingNode(true);
  return graph;
}

ge::Tensor MakeEvent(int64_t owner, int64_t type, int64_t event_seq,
                     int64_t cohort, int64_t batch,
                     const std::array<int64_t, kRows> &credits,
                     const std::array<int64_t, kRows> &seeds) {
  std::array<int64_t, kEventElements> values{};
  values[0] = owner;
  values[1] = type;
  values[2] = event_seq;
  values[3] = cohort;
  values[4] = batch;
  for (size_t row = 0; row < kRows; ++row) {
    values[5 + row] = credits[row];
    values[9 + row] = seeds[row];
  }
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(
      ge::Shape({static_cast<int64_t>(kEventElements)}), ge::FORMAT_ND,
      ge::DT_INT64));
  tensor.SetData(reinterpret_cast<uint8_t *>(values.data()),
                 values.size() * sizeof(values[0]));
  return tensor;
}

bool ParseOutput(const std::vector<ge::Tensor> &outputs,
                 std::array<int64_t, kOutputElements> &values) {
  if (outputs.size() != 1 ||
      outputs[0].GetSize() != values.size() * sizeof(values[0]) ||
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
          if (shared.stop) return;
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
    std::array<int64_t, kOutputElements> values{};
    if (!ParseOutput(outputs, values)) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      shared.fetch_status = ge::FAILED;
      shared.changed.notify_all();
      return;
    }
    std::cout << "P1_OUTPUT owner=" << values[0]
              << " type=" << values[1]
              << " event_seq=" << values[2]
              << " cohort=" << values[3]
              << " row=" << values[4]
              << " commit_seq=" << values[5]
              << " state=" << values[6]
              << " token=" << values[7]
              << " remaining=" << values[8]
              << " credit=" << values[9]
              << " status=" << values[10]
              << " reserved=" << values[11]
              << " transaction=" << flow_info.GetTransactionId()
              << " flags=" << flow_info.GetFlowFlags() << std::endl;
    {
      std::lock_guard<std::mutex> lock(shared.mutex);
      ++shared.fetch_calls;
      const RowKey key{values[3], values[4]};
      if (values[1] == kOutputCommit) {
        shared.commits[key] = values[5];
      } else if (values[1] == kOutputRetireComplete) {
        shared.retired[key] = true;
      } else if (values[1] == kOutputQuiescent) {
        ++shared.quiescent;
      } else if (values[1] == kOutputShutdown) {
        shared.aicore_calls = values[5];
        shared.total_commits = values[6];
        shared.shutdown = true;
      } else if (values[1] == kOutputRejected) {
        ++shared.rejected;
      }
      shared.changed.notify_all();
      if (shared.shutdown) return;
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
                     int64_t type, int64_t event_seq, int64_t cohort,
                     int64_t batch,
                     const std::array<int64_t, kRows> &credits,
                     const std::array<int64_t, kRows> &seeds) {
  std::vector<ge::Tensor> inputs = {
      MakeEvent(owner, type, event_seq, cohort, batch, credits, seeds)};
  ge::DataFlowInfo flow_info;
  flow_info.SetTransactionId(static_cast<uint64_t>(event_seq));
  return session->FeedDataFlowGraph(kGraphId, inputs, flow_info,
                                    kFeedTimeoutMs);
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "usage: persistent_recurrence_p1_host FUNCTION_CONFIG "
                 "GRAPH_CONFIG DEPLOY_CONFIG OWNER [placement_probe]"
              << std::endl;
    return 2;
  }
  const std::string function_config = argv[1];
  const std::string graph_config = argv[2];
  const std::string deploy_config = argv[3];
  const int64_t owner = std::stoll(argv[4]);
  const bool placement_probe =
      argc == 6 && std::string(argv[5]) == "placement_probe";
  if (access(function_config.c_str(), R_OK) != 0 ||
      access(graph_config.c_str(), R_OK) != 0 ||
      access(deploy_config.c_str(), R_OK) != 0 || owner <= 0 ||
      (argc == 6 && !placement_probe)) {
    std::cerr << "invalid configuration path, owner, or mode" << std::endl;
    return 2;
  }
  std::cout << "P1_OWNER_START pid=" << getpid() << " owner=" << owner
            << " mode=" << (placement_probe ? 1 : 0) << std::endl;
  const int64_t cpu_start_us = ProcessCpuUs();
  const auto wall_start = std::chrono::steady_clock::now();
  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.experiment.data_flow_deploy_info_path", deploy_config.c_str()},
      {"ge.graphRunMode", "0"}};
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) return ret;
  auto flow_graph = BuildFlowGraph(function_config.c_str(), graph_config.c_str(),
                                   owner);
  const auto &graph = flow_graph.ToGeGraph();
  if (!graph.IsValid()) {
    ge::GEFinalize();
    return 1;
  }
  auto session = std::make_shared<ge::Session>(config);
  ret = session->AddGraph(kGraphId, graph);
  if (ret != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return ret;
  }
  ret = session->CompileGraph(kGraphId);
  if (ret != ge::SUCCESS) {
    session->RemoveGraph(kGraphId);
    session.reset();
    ge::GEFinalize();
    return ret;
  }

  SharedState shared;
  int64_t feed_calls = 0;
  auto send = [&](int64_t type, int64_t event_seq, int64_t cohort,
                  int64_t batch,
                  const std::array<int64_t, kRows> &credits,
                  const std::array<int64_t, kRows> &seeds) -> bool {
    const auto status = FeedEvent(session, owner, type, event_seq, cohort,
                                  batch, credits, seeds);
    if (status != ge::SUCCESS) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      shared.stop = true;
      shared.changed.notify_all();
      return false;
    }
    ++feed_calls;
    return true;
  };
  const std::array<int64_t, kRows> zeros = {0, 0, 0, 0};
  std::thread drain(FetchOutputs, session, std::ref(shared));
  bool passed = false;
  int64_t blocked_row0 = 0;
  int64_t pre_shutdown_commits = 0;
  int64_t pre_shutdown_fetches = 0;
  if (placement_probe) {
    passed =
        send(kEventAdmit, 1, 1, 1, {256, 0, 0, 0}, {1000, 0, 0, 0}) &&
        WaitFor(shared, [](const SharedState &state) {
          return CommitIs(state, 1, 0, kTargetSteps) && state.quiescent == 1;
        }, 300) &&
        send(kEventShutdown, 2, 0, 0, zeros, zeros) &&
        WaitFor(shared, [](const SharedState &state) { return state.shutdown; },
                120);
  } else {
    passed =
        send(kEventAdmit, 1, 1, 1, {256, 0, 0, 0}, {1000, 0, 0, 0}) &&
        WaitFor(shared, [](const SharedState &state) {
          return CommitIs(state, 1, 0, kTargetSteps) &&
                 IsRetired(state, 1, 0) && state.quiescent == 1;
        }, 300) &&
        send(kEventAdmit, 2, 2, 4, {16, 256, 256, 256},
             {2000, 3000, 4000, 5000}) &&
        WaitFor(shared, [](const SharedState &state) {
          return CommitIs(state, 2, 0, 16) &&
                 CommitIs(state, 2, 1, kTargetSteps) &&
                 CommitIs(state, 2, 2, kTargetSteps) &&
                 CommitIs(state, 2, 3, kTargetSteps) &&
                 IsRetired(state, 2, 1) && IsRetired(state, 2, 2) &&
                 IsRetired(state, 2, 3);
        }, 300);
    if (passed) {
      {
        std::lock_guard<std::mutex> lock(shared.mutex);
        blocked_row0 = shared.commits[{2, 0}];
      }
      std::cout << "P1_CREDIT_ISOLATION owner=" << owner
                << " cohort=2 blocked_row=0 blocked_commits=" << blocked_row0
                << " peer1_commits=256 peer2_commits=256 peer3_commits=256"
                << std::endl;
      passed = send(kEventCredit, 3, 2, 0, {240, 0, 0, 0}, zeros) &&
               WaitFor(shared, [](const SharedState &state) {
                 return CommitIs(state, 2, 0, kTargetSteps) &&
                        IsRetired(state, 2, 0) && state.quiescent == 2;
               }, 300);
    }
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      pre_shutdown_commits = shared.commits[{1, 0}];
      for (int64_t row = 0; row < static_cast<int64_t>(kRows); ++row) {
        pre_shutdown_commits += shared.commits[{2, row}];
      }
      pre_shutdown_fetches = shared.fetch_calls;
      std::cout << "P1_PRE_SHUTDOWN owner=" << owner
                << " commits=" << pre_shutdown_commits
                << " fetch_calls=" << pre_shutdown_fetches << std::endl;
    }
    passed = passed && send(kEventShutdown, 4, 0, 0, zeros, zeros) &&
             WaitFor(shared,
                     [](const SharedState &state) { return state.shutdown; },
                     120);
  }
  if (drain.joinable()) drain.join();
  const auto remove_status = session->RemoveGraph(kGraphId);
  session.reset();
  const auto finalize_status = ge::GEFinalize();
  const int64_t cpu_end_us = ProcessCpuUs();
  const auto wall_end = std::chrono::steady_clock::now();
  int64_t fetch_calls = 0;
  int64_t quiescent = 0;
  int64_t rejected = 0;
  int64_t aicore_calls = 0;
  int64_t total_commits = 0;
  int64_t b1_row0_commits = 0;
  std::array<int64_t, kRows> b4_commits{};
  int32_t fetch_status = ge::FAILED;
  {
    std::lock_guard<std::mutex> lock(shared.mutex);
    fetch_calls = shared.fetch_calls;
    quiescent = shared.quiescent;
    rejected = shared.rejected;
    aicore_calls = shared.aicore_calls;
    total_commits = shared.total_commits;
    b1_row0_commits = shared.commits[{1, 0}];
    for (size_t row = 0; row < kRows; ++row) {
      b4_commits[row] = shared.commits[{2, static_cast<int64_t>(row)}];
    }
    fetch_status = shared.fetch_status;
  }
  const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           wall_end - wall_start)
                           .count();
  if (placement_probe) {
    std::cout << "P1_PLACEMENT_PROBE owner=" << owner
              << " feed_calls=" << feed_calls
              << " commits=" << total_commits
              << " aicore_calls=" << aicore_calls
              << " quiescent=" << quiescent
              << " compile_status=0 fetch_status=" << fetch_status
              << " remove_status=" << remove_status
              << " finalize_status=" << finalize_status << std::endl;
    return passed && feed_calls == 2 && total_commits == 256 &&
                   aicore_calls == 256 && quiescent == 1 && rejected == 0 &&
                   fetch_status == ge::SUCCESS && remove_status == ge::SUCCESS &&
                   finalize_status == ge::SUCCESS
               ? 0
               : 1;
  }
  std::cout << "P1_SUMMARY owner=" << owner
            << " owner_starts=1 feed_calls=" << feed_calls
            << " fetch_calls=" << fetch_calls
            << " admission_events=2 credit_events=1 shutdown_events=1"
            << " quiescent=" << quiescent << " rejected=" << rejected
            << " b1_row0_commits=" << b1_row0_commits
            << " b4_row0_commits=" << b4_commits[0]
            << " b4_row1_commits=" << b4_commits[1]
            << " b4_row2_commits=" << b4_commits[2]
            << " b4_row3_commits=" << b4_commits[3]
            << " blocked_row0_commits=" << blocked_row0
            << " aicore_calls=" << aicore_calls
            << " total_commits=" << total_commits
            << " pre_shutdown_commits=" << pre_shutdown_commits
            << " pre_shutdown_fetches=" << pre_shutdown_fetches
            << " host_cpu_us=" << (cpu_end_us - cpu_start_us)
            << " wall_ms=" << wall_ms
            << " compile_status=0 fetch_status=" << fetch_status
            << " remove_status=" << remove_status
            << " finalize_status=" << finalize_status << std::endl;
  return passed && feed_calls == 4 && quiescent == 2 && rejected == 0 &&
                 blocked_row0 == 16 && aicore_calls == kExpectedAicoreCalls &&
                 total_commits == kExpectedCommits &&
                 pre_shutdown_commits == kExpectedCommits &&
                 pre_shutdown_fetches > 0 && fetch_status == ge::SUCCESS &&
                 remove_status == ge::SUCCESS && finalize_status == ge::SUCCESS
             ? 0
             : 1;
}
