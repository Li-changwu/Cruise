#include <algorithm>
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
constexpr size_t kEventElements = 12;
constexpr size_t kOutputElements = 16;
constexpr int32_t kFeedTimeoutMs = 30000;
constexpr int32_t kFetchTimeoutMs = 10000;
constexpr int32_t kFetchStartupRetryDelayMs = 100;
constexpr int32_t kFetchStartupRetryLimit = 1200;
constexpr int64_t kEventAdmit = 1;
constexpr int64_t kEventCredit = 2;
constexpr int64_t kEventCancel = 3;
constexpr int64_t kEventShutdown = 4;
constexpr int64_t kOutputAdmitAck = 1;
constexpr int64_t kOutputCommit = 2;
constexpr int64_t kOutputRetireComplete = 3;
constexpr int64_t kOutputRetireCancelled = 4;
constexpr int64_t kOutputCreditAck = 5;
constexpr int64_t kOutputQuiescent = 6;
constexpr int64_t kOutputCumulativeAck = 7;
constexpr int64_t kOutputRejected = 8;
constexpr int64_t kOutputShutdown = 9;
constexpr int64_t kStatusWrongOwner = 1;
constexpr int64_t kStatusStaleEvent = 2;
constexpr int64_t kStatusNoCapacity = 3;
constexpr int64_t kStatusBadGeneration = 4;
constexpr int64_t kStatusNotActive = 9;
constexpr int64_t kStressCohorts = 250;
constexpr int64_t kStressCancellations = kStressCohorts * kRows;
constexpr int64_t kExpectedMatrixCancelled = 8;
constexpr int64_t kExpectedRetired = kStressCancellations + 9;
constexpr int64_t kExpectedFeedCalls = 2028;
constexpr int64_t kExpectedDuplicateAcks = 4;
constexpr int64_t kExpectedRejected = 5;
constexpr int64_t kExpectedQuiescent = 255;

using RequestKey = std::pair<int64_t, int64_t>;

struct Event {
  int64_t owner = 0;
  int64_t type = 0;
  int64_t seq = 0;
  int64_t request = 0;
  int64_t generation = 0;
  int64_t row = -1;
  int64_t target_steps = 0;
  int64_t credit = 0;
  int64_t cancel_seq = 0;
  int64_t seed = 0;
  int64_t flags = 0;
  int64_t reserved = 0;
};

struct RequestState {
  int64_t row = -1;
  int64_t commits = 0;
  bool admitted = false;
  bool retired = false;
  bool cancelled = false;
};

struct CancelSent {
  std::chrono::steady_clock::time_point time;
  bool stress = false;
};

struct SharedState {
  std::mutex mutex;
  std::condition_variable changed;
  std::map<RequestKey, RequestState> requests;
  std::map<int64_t, int64_t> duplicate_acks;
  std::map<int64_t, std::vector<int64_t>> rejected_statuses;
  std::map<int64_t, int64_t> credit_acks;
  std::map<int64_t, CancelSent> cancel_sent;
  std::vector<int64_t> stress_cancel_latency_us;
  int64_t fetch_calls = 0;
  int64_t quiescent = 0;
  int64_t aicore_calls = 0;
  int64_t total_commits = 0;
  int64_t total_retired = 0;
  bool shutdown = false;
  bool stop = false;
  int32_t fetch_status = ge::SUCCESS;
};

Event Admit(int64_t owner, int64_t seq, int64_t request, int64_t generation,
            int64_t target_steps, int64_t credit, int64_t seed) {
  Event event;
  event.owner = owner;
  event.type = kEventAdmit;
  event.seq = seq;
  event.request = request;
  event.generation = generation;
  event.target_steps = target_steps;
  event.credit = credit;
  event.seed = seed;
  return event;
}

Event Credit(int64_t owner, int64_t seq, int64_t request, int64_t generation,
             int64_t row, int64_t credit) {
  Event event;
  event.owner = owner;
  event.type = kEventCredit;
  event.seq = seq;
  event.request = request;
  event.generation = generation;
  event.row = row;
  event.credit = credit;
  return event;
}

Event Cancel(int64_t owner, int64_t seq, int64_t request,
             int64_t generation, int64_t row, int64_t cancel_seq) {
  Event event;
  event.owner = owner;
  event.type = kEventCancel;
  event.seq = seq;
  event.request = request;
  event.generation = generation;
  event.row = row;
  event.cancel_seq = cancel_seq;
  return event;
}

Event Shutdown(int64_t owner, int64_t seq) {
  Event event;
  event.owner = owner;
  event.type = kEventShutdown;
  event.seq = seq;
  return event;
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
  auto add = ge::op::Add("persistent_cancel_p2_add")
                 .set_input_x1(state)
                 .set_input_x2(delta);
  ge::Graph graph("PersistentCancelP2AddGraph");
  graph.SetInputs({state, delta}).SetOutputs({add});
  return graph;
}

ge::dflow::FlowGraph BuildFlowGraph(const char *function_config,
                                    const char *graph_config,
                                    int64_t owner) {
  using namespace ge::dflow;
  auto event = FlowData("Event", 0);
  auto recurrence =
      GraphPp("persistent_cancel_p2_graph_pp", BuildRecurrenceGraph)
          .SetCompileConfig(graph_config);
  auto controller = FunctionPp("persistent_cancel_p2_pp")
                        .SetCompileConfig(function_config)
                        .SetInitParam("owner_instance_id", owner);
  controller.AddInvokedClosure("recurrence_graph", recurrence);
  auto node = FlowNode("persistent_cancel_p2_node", 1, 1);
  node.AddPp(controller).SetInput(0, event);
  FlowGraph graph("cruise_persistent_cancel_p2");
  std::vector<FlowOperator> inputs = {event};
  std::vector<std::pair<FlowOperator, std::vector<size_t>>> outputs = {
      {node, {0}}};
  graph.SetInputs(inputs).SetOutputs(outputs).SetContainsNMappingNode(true);
  return graph;
}

ge::Tensor MakeEvent(const Event &event) {
  const std::array<int64_t, kEventElements> values = {
      event.owner,        event.type,       event.seq,
      event.request,      event.generation, event.row,
      event.target_steps, event.credit,     event.cancel_seq,
      event.seed,         event.flags,      event.reserved};
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(
      ge::Shape({static_cast<int64_t>(kEventElements)}), ge::FORMAT_ND,
      ge::DT_INT64));
  tensor.SetData(reinterpret_cast<const uint8_t *>(values.data()),
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
      std::lock_guard<std::mutex> lock(shared.mutex);
      if (!fetched_output && !shared.stop &&
          startup_failures < kFetchStartupRetryLimit) {
        ++startup_failures;
      } else {
        shared.fetch_status = ret;
        shared.changed.notify_all();
        return;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kFetchStartupRetryDelayMs));
      continue;
    }
    fetched_output = true;
    std::array<int64_t, kOutputElements> values{};
    if (!ParseOutput(outputs, values)) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      shared.fetch_status = ge::FAILED;
      shared.changed.notify_all();
      return;
    }
    const auto observed = std::chrono::steady_clock::now();
    int64_t cancel_latency_us = -1;
    {
      std::lock_guard<std::mutex> lock(shared.mutex);
      ++shared.fetch_calls;
      const RequestKey key{values[3], values[4]};
      auto &request = shared.requests[key];
      if (values[1] == kOutputAdmitAck) {
        request.row = values[5];
        request.commits = 0;
        request.admitted = true;
      } else if (values[1] == kOutputCommit) {
        request.row = values[5];
        request.commits = values[6];
      } else if (values[1] == kOutputRetireComplete ||
                 values[1] == kOutputRetireCancelled) {
        request.row = values[5];
        request.commits = values[6];
        request.retired = true;
        request.cancelled = values[1] == kOutputRetireCancelled;
        if (values[1] == kOutputRetireCancelled) {
          const auto sent = shared.cancel_sent.find(values[2]);
          if (sent != shared.cancel_sent.end()) {
            cancel_latency_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    observed - sent->second.time)
                    .count();
            if (sent->second.stress) {
              shared.stress_cancel_latency_us.push_back(cancel_latency_us);
            }
            shared.cancel_sent.erase(sent);
          }
        }
      } else if (values[1] == kOutputCreditAck) {
        ++shared.credit_acks[values[2]];
      } else if (values[1] == kOutputQuiescent) {
        ++shared.quiescent;
      } else if (values[1] == kOutputCumulativeAck) {
        ++shared.duplicate_acks[values[2]];
      } else if (values[1] == kOutputRejected) {
        shared.rejected_statuses[values[2]].push_back(values[11]);
      } else if (values[1] == kOutputShutdown) {
        shared.aicore_calls = values[12];
        shared.total_commits = values[13];
        shared.total_retired = values[14];
        shared.shutdown = true;
      }
      std::cout << "P2_OUTPUT owner=" << values[0]
                << " type=" << values[1]
                << " event_seq=" << values[2]
                << " request=" << values[3]
                << " generation=" << values[4]
                << " row=" << values[5]
                << " commit_seq=" << values[6]
                << " state=" << values[7]
                << " remaining=" << values[8]
                << " credit=" << values[9]
                << " cancel_seq=" << values[10]
                << " status=" << values[11]
                << " aicore_calls=" << values[12]
                << " total_commits=" << values[13]
                << " total_retired=" << values[14]
                << " quiescent=" << values[15]
                << " host_cancel_latency_us=" << cancel_latency_us
                << " transaction=" << flow_info.GetTransactionId()
                << " flags=" << flow_info.GetFlowFlags() << std::endl;
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

bool HasRejected(const SharedState &state, int64_t event_seq, int64_t status) {
  const auto found = state.rejected_statuses.find(event_seq);
  if (found == state.rejected_statuses.end()) return false;
  return std::find(found->second.begin(), found->second.end(), status) !=
         found->second.end();
}

bool AllAdmittedAndCommitted(const SharedState &state,
                             const std::vector<RequestKey> &keys,
                             int64_t commits) {
  for (const auto &key : keys) {
    const auto found = state.requests.find(key);
    if (found == state.requests.end() || !found->second.admitted ||
        found->second.commits < commits) {
      return false;
    }
  }
  return true;
}

bool AllRetired(const SharedState &state,
                const std::vector<RequestKey> &keys) {
  for (const auto &key : keys) {
    const auto found = state.requests.find(key);
    if (found == state.requests.end() || !found->second.retired) return false;
  }
  return true;
}

ge::Status FeedEvent(const std::shared_ptr<ge::Session> &session,
                     const Event &event, uint64_t transport_seq) {
  std::vector<ge::Tensor> inputs = {MakeEvent(event)};
  ge::DataFlowInfo flow_info;
  flow_info.SetTransactionId(transport_seq);
  return session->FeedDataFlowGraph(kGraphId, inputs, flow_info,
                                    kFeedTimeoutMs);
}

int64_t Percentile(std::vector<int64_t> values, int percent) {
  if (values.empty() || percent <= 0 || percent > 100) return -1;
  std::sort(values.begin(), values.end());
  const size_t rank =
      (static_cast<size_t>(percent) * values.size() + 99) / 100;
  return values[rank - 1];
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "usage: persistent_cancel_p2_host FUNCTION_CONFIG "
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

  std::cout << "P2_OWNER_START pid=" << getpid() << " owner=" << owner
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
  auto flow_graph =
      BuildFlowGraph(function_config.c_str(), graph_config.c_str(), owner);
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
  uint64_t transport_seq = 0;
  auto send = [&](const Event &event, bool track_stress_cancel = false) -> bool {
    if (track_stress_cancel) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      auto &sent = shared.cancel_sent[event.seq];
      sent.time = std::chrono::steady_clock::now();
      sent.stress = true;
    }
    const auto status = FeedEvent(session, event, ++transport_seq);
    if (status != ge::SUCCESS) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      shared.cancel_sent.erase(event.seq);
      shared.stop = true;
      shared.changed.notify_all();
      return false;
    }
    ++feed_calls;
    return true;
  };

  std::thread drain(FetchOutputs, session, std::ref(shared));
  bool passed = true;
  int64_t before_first_commits = -1;
  int64_t during_commits = -1;
  int64_t completed_commits = -1;
  int64_t blocked_commits = -1;
  int64_t reused_row = -1;
  bool stale_generation_isolated = false;
  bool retirement_before_reuse = false;

  if (placement_probe) {
    const RequestKey key{1, 1};
    passed = send(Admit(owner, 1, key.first, key.second, 1, 1, 1000)) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(key);
                       return found != state.requests.end() &&
                              found->second.retired;
                     },
                     120) &&
             send(Shutdown(owner, 2)) &&
             WaitFor(shared,
                     [](const SharedState &state) { return state.shutdown; },
                     120);
  } else {
    const RequestKey before_first{101, 1};
    Event admit_before = Admit(owner, 1, 101, 1, 8, 0, 1000);
    passed = send(admit_before) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(before_first);
                       return found != state.requests.end() &&
                              found->second.admitted;
                     },
                     120) &&
             send(admit_before) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       const auto found = state.duplicate_acks.find(1);
                       return found != state.duplicate_acks.end() &&
                              found->second == 1;
                     },
                     120);
    int64_t row_before = -1;
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      row_before = shared.requests[before_first].row;
    }
    Event cancel_before = Cancel(owner, 2, 101, 1, row_before, 1);
    passed = passed && send(cancel_before) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(before_first);
                       return found != state.requests.end() &&
                              found->second.retired &&
                              found->second.cancelled;
                     },
                     120);
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      before_first_commits = shared.requests[before_first].commits;
    }
    passed = passed && send(cancel_before) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       const auto found = state.duplicate_acks.find(2);
                       return found != state.duplicate_acks.end() &&
                              found->second == 1;
                     },
                     120);

    const RequestKey completed{102, 2};
    Event admit_completed = Admit(owner, 3, 102, 2, 4, 4, 2000);
    passed = passed && send(admit_completed) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(completed);
                       return found != state.requests.end() &&
                              found->second.retired &&
                              !found->second.cancelled;
                     },
                     120);
    int64_t completed_row = -1;
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      completed_commits = shared.requests[completed].commits;
      completed_row = shared.requests[completed].row;
    }
    passed = passed && send(Cancel(owner, 4, 102, 2, completed_row, 1)) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       return HasRejected(state, 4, kStatusNotActive);
                     },
                     120) &&
             send(admit_completed) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       return HasRejected(state, 3, kStatusStaleEvent);
                     },
                     120) &&
             send(Cancel(owner + 999, 5, 102, 2, completed_row, 2)) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       return HasRejected(state, 5, kStatusWrongOwner);
                     },
                     120);

    const RequestKey during{103, 3};
    passed = passed && send(Admit(owner, 6, 103, 3, 256, 256, 3000)) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(during);
                       return found != state.requests.end() &&
                              found->second.commits >= 32;
                     },
                     120);
    int64_t during_row = -1;
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      during_row = shared.requests[during].row;
    }
    Event cancel_during = Cancel(owner, 7, 103, 3, during_row, 1);
    passed = passed && send(cancel_during) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(during);
                       return found != state.requests.end() &&
                              found->second.retired;
                     },
                     120);
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      during_commits = shared.requests[during].commits;
    }
    passed = passed && send(cancel_during) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       const auto found = state.duplicate_acks.find(7);
                       return found != state.duplicate_acks.end() &&
                              found->second == 1;
                     },
                     120);

    const RequestKey isolated{104, 4};
    passed = passed && send(Admit(owner, 8, 104, 4, 64, 16, 4000)) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(isolated);
                       return found != state.requests.end() &&
                              found->second.admitted;
                     },
                     120);
    int64_t isolated_row = -1;
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      isolated_row = shared.requests[isolated].row;
    }
    passed = passed &&
             send(Cancel(owner, 9, 103, 3, isolated_row, 2)) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       return HasRejected(state, 9, kStatusBadGeneration);
                     },
                     120) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(isolated);
                       return found != state.requests.end() &&
                              found->second.commits == 16;
                     },
                     120) &&
             send(Credit(owner, 10, 104, 4, isolated_row, 48)) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       const auto found = state.credit_acks.find(10);
                       return found != state.credit_acks.end() &&
                              found->second == 1;
                     },
                     120) &&
             send(Credit(owner, 10, 104, 4, isolated_row, 48)) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       const auto found = state.duplicate_acks.find(10);
                       return found != state.duplicate_acks.end() &&
                              found->second == 1;
                     },
                     120) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(isolated);
                       return found != state.requests.end() &&
                              found->second.commits >= 20;
                     },
                     120) &&
             send(Cancel(owner, 11, 104, 4, isolated_row, 1)) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(isolated);
                       return found != state.requests.end() &&
                              found->second.retired;
                     },
                     120);
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      blocked_commits = 16;
      stale_generation_isolated =
          shared.requests[isolated].commits >= 20 &&
          shared.requests[isolated].cancelled;
    }

    std::vector<RequestKey> capacity_keys;
    for (int64_t row = 0; row < static_cast<int64_t>(kRows); ++row) {
      const RequestKey key{201 + row, 10 + row};
      capacity_keys.push_back(key);
      passed = passed && send(Admit(owner, 12 + row, key.first, key.second,
                                    256, 256, 5000 + row * 100));
    }
    passed = passed &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       return AllAdmittedAndCommitted(state, capacity_keys, 1);
                     },
                     120) &&
             send(Admit(owner, 16, 205, 14, 256, 256, 6000)) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       return HasRejected(state, 16, kStatusNoCapacity);
                     },
                     120);
    int64_t retired_row = -1;
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      retired_row = shared.requests[capacity_keys[0]].row;
    }
    passed = passed && send(Cancel(owner, 17, 201, 10, retired_row, 1)) &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       const auto found = state.requests.find(capacity_keys[0]);
                       return found != state.requests.end() &&
                              found->second.retired;
                     },
                     120) &&
             send(Admit(owner, 18, 205, 14, 256, 256, 6000)) &&
             WaitFor(shared,
                     [](const SharedState &state) {
                       const auto found = state.requests.find({205, 14});
                       return found != state.requests.end() &&
                              found->second.admitted;
                     },
                     120);
    if (passed) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      reused_row = shared.requests[{205, 14}].row;
      retirement_before_reuse = reused_row == retired_row;
    }
    const std::vector<RequestKey> remaining_capacity = {
        {205, 14}, capacity_keys[1], capacity_keys[2], capacity_keys[3]};
    for (size_t index = 0; index < remaining_capacity.size(); ++index) {
      int64_t row = -1;
      if (passed) {
        std::lock_guard<std::mutex> lock(shared.mutex);
        row = shared.requests[remaining_capacity[index]].row;
      }
      passed = passed && send(Cancel(owner, 19 + index,
                                     remaining_capacity[index].first,
                                     remaining_capacity[index].second, row, 1));
    }
    passed = passed &&
             WaitFor(shared,
                     [&](const SharedState &state) {
                       return AllRetired(state, remaining_capacity) &&
                              state.quiescent >= 5;
                     },
                     120);

    int64_t next_event_seq = 23;
    int64_t next_request = 10000;
    int64_t next_generation = 1000;
    for (int64_t cohort = 0; passed && cohort < kStressCohorts; ++cohort) {
      std::vector<RequestKey> keys;
      for (int64_t row = 0; row < static_cast<int64_t>(kRows); ++row) {
        const RequestKey key{next_request++, next_generation++};
        keys.push_back(key);
        passed = send(Admit(owner, next_event_seq++, key.first, key.second,
                            256, 256, 100000 + key.second));
        if (!passed) break;
      }
      passed = passed &&
               WaitFor(shared,
                       [&](const SharedState &state) {
                         return AllAdmittedAndCommitted(state, keys, 1);
                       },
                       120);
      for (const auto &key : keys) {
        int64_t row = -1;
        if (passed) {
          std::lock_guard<std::mutex> lock(shared.mutex);
          row = shared.requests[key].row;
        }
        passed = passed &&
                 send(Cancel(owner, next_event_seq++, key.first, key.second,
                             row, 1),
                      true);
      }
      passed = passed &&
               WaitFor(shared,
                       [&](const SharedState &state) {
                         return AllRetired(state, keys);
                       },
                       120);
    }
    passed = passed && send(Shutdown(owner, next_event_seq)) &&
             WaitFor(shared,
                     [](const SharedState &state) { return state.shutdown; },
                     120);
  }

  if (!passed) {
    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.stop = true;
    shared.changed.notify_all();
  }
  if (drain.joinable()) drain.join();
  const auto remove_status = session->RemoveGraph(kGraphId);
  session.reset();
  const auto finalize_status = ge::GEFinalize();
  const int64_t cpu_end_us = ProcessCpuUs();
  const auto wall_end = std::chrono::steady_clock::now();

  int64_t fetch_calls = 0;
  int64_t quiescent = 0;
  int64_t duplicate_acks = 0;
  int64_t rejected = 0;
  int64_t aicore_calls = 0;
  int64_t total_commits = 0;
  int64_t total_retired = 0;
  int64_t latency_samples = 0;
  int32_t fetch_status = ge::FAILED;
  std::vector<int64_t> latencies;
  {
    std::lock_guard<std::mutex> lock(shared.mutex);
    fetch_calls = shared.fetch_calls;
    quiescent = shared.quiescent;
    for (const auto &item : shared.duplicate_acks) {
      duplicate_acks += item.second;
    }
    for (const auto &item : shared.rejected_statuses) {
      rejected += static_cast<int64_t>(item.second.size());
    }
    aicore_calls = shared.aicore_calls;
    total_commits = shared.total_commits;
    total_retired = shared.total_retired;
    latencies = shared.stress_cancel_latency_us;
    latency_samples = static_cast<int64_t>(latencies.size());
    fetch_status = shared.fetch_status;
  }
  const int64_t latency_p50_us = Percentile(latencies, 50);
  const int64_t latency_p95_us = Percentile(latencies, 95);
  const int64_t latency_p99_us = Percentile(latencies, 99);
  const int64_t latency_max_us =
      latencies.empty() ? -1 : *std::max_element(latencies.begin(), latencies.end());
  const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           wall_end - wall_start)
                           .count();

  if (placement_probe) {
    std::cout << "P2_PLACEMENT_PROBE owner=" << owner
              << " feed_calls=" << feed_calls
              << " commits=" << total_commits
              << " aicore_calls=" << aicore_calls
              << " retired=" << total_retired
              << " quiescent=" << quiescent
              << " compile_status=0 fetch_status=" << fetch_status
              << " remove_status=" << remove_status
              << " finalize_status=" << finalize_status << std::endl;
    return passed && feed_calls == 2 && total_commits == 1 &&
                   aicore_calls == 1 && total_retired == 1 &&
                   quiescent == 1 && fetch_status == ge::SUCCESS &&
                   remove_status == ge::SUCCESS &&
                   finalize_status == ge::SUCCESS
               ? 0
               : 1;
  }

  std::cout << "P2_MATRIX owner=" << owner
            << " before_first_commits=" << before_first_commits
            << " during_commits=" << during_commits
            << " completed_commits=" << completed_commits
            << " blocked_commits=" << blocked_commits
            << " reused_row=" << reused_row
            << " duplicate_acks=" << duplicate_acks
            << " rejected=" << rejected
            << " matrix_cancelled=" << kExpectedMatrixCancelled
            << " retirement_before_reuse=" << (retirement_before_reuse ? 1 : 0)
            << " stale_generation_isolated="
            << (stale_generation_isolated ? 1 : 0) << std::endl;
  std::cout << "P2_CANCEL_LATENCY owner=" << owner
            << " samples=" << latency_samples
            << " p50_us=" << latency_p50_us
            << " p95_us=" << latency_p95_us
            << " p99_us=" << latency_p99_us
            << " max_us=" << latency_max_us << std::endl;
  std::cout << "P2_SUMMARY owner=" << owner
            << " owner_starts=1 feed_calls=" << feed_calls
            << " fetch_calls=" << fetch_calls
            << " matrix_admissions=9 matrix_cancelled=8"
            << " stress_admissions=1000 stress_cancelled=1000"
            << " duplicate_acks=" << duplicate_acks
            << " rejected=" << rejected
            << " quiescent=" << quiescent
            << " aicore_calls=" << aicore_calls
            << " total_commits=" << total_commits
            << " total_retired=" << total_retired
            << " latency_samples=" << latency_samples
            << " latency_p95_us=" << latency_p95_us
            << " latency_p99_us=" << latency_p99_us
            << " host_cpu_us=" << (cpu_end_us - cpu_start_us)
            << " wall_ms=" << wall_ms
            << " compile_status=0 fetch_status=" << fetch_status
            << " remove_status=" << remove_status
            << " finalize_status=" << finalize_status << std::endl;

  return passed && before_first_commits == 0 && completed_commits == 4 &&
                 during_commits >= 32 && during_commits < 256 &&
                 blocked_commits == 16 && retirement_before_reuse &&
                 stale_generation_isolated &&
                 feed_calls == kExpectedFeedCalls &&
                 duplicate_acks == kExpectedDuplicateAcks &&
                 rejected == kExpectedRejected &&
                 quiescent == kExpectedQuiescent &&
                 total_retired == kExpectedRetired &&
                 latency_samples == kStressCancellations &&
                 latency_p95_us >= 0 && latency_p95_us <= 50000 &&
                 latency_p99_us >= 0 && latency_p99_us <= 100000 &&
                 fetch_status == ge::SUCCESS && remove_status == ge::SUCCESS &&
                 finalize_status == ge::SUCCESS
             ? 0
             : 1;
}
