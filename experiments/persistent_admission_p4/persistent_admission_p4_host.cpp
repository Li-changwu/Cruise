#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <functional>
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
#include "graph/graph.h"

namespace {
constexpr uint32_t kGraphId = 0;
constexpr int32_t kRows = 4;
constexpr int32_t kEventHeaderElements = 16;
constexpr int32_t kEventElements = 144;
constexpr int32_t kOutputElements = 20;
constexpr int32_t kFeedTimeoutMs = 30000;
constexpr int32_t kFetchTimeoutMs = 10000;
constexpr int32_t kStartupRetryDelayMs = 100;
constexpr int32_t kStartupRetryLimit = 1200;
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
constexpr int64_t kStatusNoCapacity = 3;
constexpr int64_t kPrimaryRequests = 32;
constexpr int64_t kPrimaryConcurrency = 4;

using RequestKey = std::pair<int64_t, int64_t>;

int64_t SteadyNanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Event {
  int64_t owner = 0;
  int64_t type = 0;
  int64_t seq = 0;
  int64_t request = 0;
  int64_t generation = 0;
  int64_t row = -1;
  int64_t prompt_len = 0;
  int64_t target_steps = 0;
  int64_t credit = 0;
  int64_t cancel_seq = 0;
  int64_t eos_token = 151645;
  int64_t flags = 0;
  std::array<int64_t, 4> reserved{};
  std::array<int64_t, 128> prompt{};
};

struct RequestSpec {
  RequestKey key;
  std::string tag;
  std::vector<int64_t> prompt;
  int64_t target_steps = 0;
  int64_t flags = 0;
  int64_t initial_credit = -1;
  int64_t expected_commits = 0;
  int64_t expected_finish_reason = 0;
  bool expected_cancelled = false;
};

struct RequestState {
  std::string tag;
  int64_t row = -1;
  int64_t commits = 0;
  int64_t send_count = 0;
  int64_t rejected_admissions = 0;
  int64_t stream_messages = 0;
  bool admitted = false;
  bool retired = false;
  bool cancelled = false;
  int64_t final_position = -1;
  int64_t final_page = -1;
  int64_t final_checksum = 0;
  int64_t finish_reason = 0;
  int64_t first_send_ns = 0;
  int64_t admit_ns = 0;
  int64_t first_commit_ns = 0;
  int64_t last_commit_ns = 0;
  int64_t retire_ns = 0;
  std::vector<int64_t> tokens;
  std::vector<int64_t> token_timestamps_ns;
};

struct SharedState {
  std::mutex mutex;
  std::condition_variable changed;
  std::map<RequestKey, RequestState> requests;
  int64_t fetch_calls = 0;
  int64_t aicore_calls = 0;
  int64_t total_commits = 0;
  int64_t total_retired = 0;
  int64_t quiescent = 0;
  int64_t rejected = 0;
  int64_t credit_acks = 0;
  int64_t duplicate_acks = 0;
  int64_t single_token_outputs = 0;
  int64_t protocol_errors = 0;
  int64_t max_active_rows = 0;
  int64_t max_pending_requests = 0;
  std::map<int64_t, int64_t> rejected_statuses;
  bool shutdown = false;
  bool stop = false;
  int32_t fetch_status = ge::SUCCESS;
};

Event Admission(int64_t owner, int64_t seq, const RequestSpec &spec,
                int64_t cohort_id, int64_t cohort_size) {
  Event event;
  event.owner = owner;
  event.type = kEventAdmit;
  event.seq = seq;
  event.request = spec.key.first;
  event.generation = spec.key.second;
  event.prompt_len = static_cast<int64_t>(spec.prompt.size());
  event.target_steps = spec.target_steps;
  event.credit = spec.initial_credit < 0 ? spec.target_steps : spec.initial_credit;
  event.flags = spec.flags;
  event.reserved[0] = cohort_id;
  event.reserved[1] = cohort_size;
  for (size_t index = 0; index < spec.prompt.size() && index < event.prompt.size();
       ++index) {
    event.prompt[index] = spec.prompt[index];
  }
  return event;
}

Event Credit(int64_t owner, int64_t seq, const RequestSpec &spec, int64_t row,
             int64_t credit) {
  Event event;
  event.owner = owner;
  event.type = kEventCredit;
  event.seq = seq;
  event.request = spec.key.first;
  event.generation = spec.key.second;
  event.row = row;
  event.credit = credit;
  return event;
}

Event Cancel(int64_t owner, int64_t seq, const RequestSpec &spec, int64_t row) {
  Event event;
  event.owner = owner;
  event.type = kEventCancel;
  event.seq = seq;
  event.request = spec.key.first;
  event.generation = spec.key.second;
  event.row = row;
  event.cancel_seq = 1;
  return event;
}

Event Shutdown(int64_t owner, int64_t seq) {
  Event event;
  event.owner = owner;
  event.type = kEventShutdown;
  event.seq = seq;
  return event;
}

ge::dflow::FlowGraph BuildFlowGraph(const std::string &air_path,
                                    const std::string &graph_config,
                                    const std::string &function_config,
                                    int64_t owner) {
  using namespace ge::dflow;
  auto event = FlowData("Event", 0);
  auto decoder = GraphPp("persistent_admission_p4_graph_pp", [air_path]() {
    ge::Graph graph("PersistentAdmissionP4AIR");
    const auto status = graph.LoadFromFile(air_path.c_str());
    std::cout << "P4_AIR_LOAD status=" << status
              << " valid=" << graph.IsValid() << std::endl;
    return graph;
  });
  decoder.SetCompileConfig(graph_config.c_str());
  auto controller = FunctionPp("persistent_decoder_p3_pp")
                        .SetCompileConfig(function_config.c_str())
                        .SetInitParam("owner_instance_id", owner);
  controller.AddInvokedClosure("decode_graph_0", decoder);
  // Keep the deployed node identity aligned with the validated P3 config.
  auto node = FlowNode("persistent_decoder_p3_node", 1, 1);
  node.AddPp(controller).SetInput(0, event);
  FlowGraph graph("cruise_persistent_admission_p4");
  graph.SetInputs({event}).SetOutputs({{node, {0}}}).SetContainsNMappingNode(true);
  return graph;
}

ge::Tensor MakeEvent(const Event &event) {
  std::array<int64_t, kEventElements> values{};
  values[0] = event.owner;
  values[1] = event.type;
  values[2] = event.seq;
  values[3] = event.request;
  values[4] = event.generation;
  values[5] = event.row;
  values[6] = event.prompt_len;
  values[7] = event.target_steps;
  values[8] = event.credit;
  values[9] = event.cancel_seq;
  values[10] = event.eos_token;
  values[11] = event.flags;
  for (size_t index = 0; index < event.reserved.size(); ++index) {
    values[12 + index] = event.reserved[index];
  }
  std::copy(event.prompt.begin(), event.prompt.end(),
            values.begin() + kEventHeaderElements);
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

int64_t CountActive(const SharedState &shared) {
  int64_t count = 0;
  for (const auto &item : shared.requests) {
    if (item.second.admitted && !item.second.retired) ++count;
  }
  return count;
}

int64_t CountPending(const SharedState &shared) {
  int64_t count = 0;
  for (const auto &item : shared.requests) {
    const auto &state = item.second;
    if (!state.retired && state.send_count > 0) ++count;
  }
  return count;
}

void FetchOutputs(const std::shared_ptr<ge::Session> &session,
                  SharedState &shared) {
  int32_t startup_failures = 0;
  bool fetched_output = false;
  while (true) {
    std::vector<ge::Tensor> outputs;
    ge::DataFlowInfo flow_info;
    const auto ret = session->FetchDataFlowGraph(kGraphId, outputs, flow_info,
                                                  kFetchTimeoutMs);
    if (ret != ge::SUCCESS) {
      bool retry = false;
      {
        std::lock_guard<std::mutex> lock(shared.mutex);
        if (!fetched_output && !shared.stop &&
            startup_failures < kStartupRetryLimit) {
          ++startup_failures;
          retry = true;
        } else {
          shared.fetch_status = ret;
          shared.changed.notify_all();
        }
      }
      if (!retry) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(kStartupRetryDelayMs));
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
    const int64_t now = SteadyNanos();
    {
      std::lock_guard<std::mutex> lock(shared.mutex);
      ++shared.fetch_calls;
      const RequestKey key{values[3], values[4]};
      auto &request = shared.requests[key];
      if (values[1] == kOutputAdmitAck) {
        request.row = values[5];
        request.admitted = true;
        request.retired = false;
        request.admit_ns = now;
        shared.max_active_rows = std::max(shared.max_active_rows, CountActive(shared));
      } else if (values[1] == kOutputCommit) {
        if (values[6] != request.commits + 1) ++shared.protocol_errors;
        request.row = values[5];
        request.commits = values[6];
        request.tokens.push_back(values[7]);
        request.token_timestamps_ns.push_back(now);
        ++request.stream_messages;
        ++shared.single_token_outputs;
        if (request.first_commit_ns == 0) request.first_commit_ns = now;
        request.last_commit_ns = now;
      } else if (values[1] == kOutputRetireComplete ||
                 values[1] == kOutputRetireCancelled) {
        request.row = values[5];
        request.commits = values[6];
        request.retired = true;
        request.cancelled = values[1] == kOutputRetireCancelled;
        request.final_position = values[8];
        request.final_page = values[9];
        request.final_checksum = values[17];
        request.finish_reason = values[18];
        request.retire_ns = now;
      } else if (values[1] == kOutputCreditAck) {
        ++shared.credit_acks;
      } else if (values[1] == kOutputQuiescent) {
        ++shared.quiescent;
      } else if (values[1] == kOutputRejected) {
        ++shared.rejected;
        ++shared.rejected_statuses[values[13]];
        if (values[1] == kOutputRejected && values[3] > 0) {
          ++request.rejected_admissions;
        }
      } else if (values[1] == kOutputCumulativeAck) {
        ++shared.duplicate_acks;
      } else if (values[1] == kOutputShutdown) {
        shared.aicore_calls = values[14];
        shared.total_commits = values[15];
        shared.total_retired = values[16];
        shared.shutdown = true;
      }
      std::cout << "P4_OUTPUT type=" << values[1]
                << " event_seq=" << values[2]
                << " request=" << values[3]
                << " generation=" << values[4]
                << " row=" << values[5]
                << " commit_seq=" << values[6]
                << " token=" << values[7]
                << " position=" << values[8]
                << " page=" << values[9]
                << " status=" << values[13]
                << " aicore_calls=" << values[14]
                << " total_commits=" << values[15]
                << " total_retired=" << values[16]
                << " checksum=" << values[17]
                << " finish_reason=" << values[18]
                << " transaction=" << flow_info.GetTransactionId()
                << std::endl;
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

ge::Status FeedEvent(const std::shared_ptr<ge::Session> &session,
                     const Event &event, uint64_t transport_seq) {
  ge::DataFlowInfo flow_info;
  flow_info.SetTransactionId(transport_seq);
  return session->FeedDataFlowGraph(kGraphId, {MakeEvent(event)}, flow_info,
                                    kFeedTimeoutMs);
}

size_t RetiredInRange(const SharedState &shared,
                      const std::vector<RequestSpec> &specs, size_t begin,
                      size_t end) {
  size_t count = 0;
  for (size_t index = begin; index < end; ++index) {
    const auto found = shared.requests.find(specs[index].key);
    if (found != shared.requests.end() && found->second.retired) ++count;
  }
  return count;
}

size_t AdmittedWithCommitsInRange(const SharedState &shared,
                                  const std::vector<RequestSpec> &specs,
                                  size_t begin, size_t end, int64_t commits) {
  size_t count = 0;
  for (size_t index = begin; index < end; ++index) {
    const auto found = shared.requests.find(specs[index].key);
    if (found != shared.requests.end() && found->second.admitted &&
        found->second.commits >= commits) {
      ++count;
    }
  }
  return count;
}

std::vector<int64_t> FullPrompt() {
  std::vector<int64_t> prompt(128);
  for (int64_t index = 0; index < 128; ++index) prompt[index] = 1000 + index;
  return prompt;
}

std::vector<int64_t> ShortPrompt() { return {9707, 9708, 9709, 9710}; }

std::vector<int64_t> EosPrompt() {
  return {151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553,
          54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645,
          198, 151644, 872, 198, 5598, 458, 4287, 2033, 13, 3155,
          537, 2550, 894, 1467, 13, 151645, 198, 151644, 77091, 198};
}

RequestSpec MakeSpec(size_t index, const std::string &tag,
                     const std::vector<int64_t> &prompt, int64_t target,
                     int64_t flags, int64_t expected_commits,
                     int64_t expected_finish, int64_t initial_credit = -1,
                     bool cancelled = false) {
  RequestSpec spec;
  spec.key = {20000 + static_cast<int64_t>(index),
              1 + static_cast<int64_t>(index)};
  spec.tag = tag;
  spec.prompt = prompt;
  spec.target_steps = target;
  spec.flags = flags;
  spec.initial_credit = initial_credit;
  spec.expected_commits = expected_commits;
  spec.expected_finish_reason = expected_finish;
  spec.expected_cancelled = cancelled;
  return spec;
}

std::vector<RequestSpec> BuildSpecs(const std::string &scenario) {
  std::vector<RequestSpec> specs;
  if (scenario == "primary-c4") {
    const auto prompt = FullPrompt();
    for (size_t index = 0; index < kPrimaryRequests; ++index) {
      specs.push_back(MakeSpec(index, "primary", prompt, 256, 1, 256, 2));
    }
    return specs;
  }
  if (scenario != "regression") return specs;
  const auto short_prompt = ShortPrompt();
  size_t index = 0;
  for (size_t count = 0; count < 8; ++count, ++index) {
    specs.push_back(MakeSpec(index, "short", short_prompt, 8, 1, 8, 2));
  }
  specs.push_back(MakeSpec(index++, "eos", EosPrompt(), 8, 0, 1, 1));
  for (size_t count = 0; count < 8; ++count, ++index) {
    specs.push_back(MakeSpec(index, "burst", short_prompt, 8, 1, 8, 2));
  }
  for (size_t count = 0; count < 8; ++count, ++index) {
    const int64_t credit = count < 4 ? 1 : -1;
    specs.push_back(
        MakeSpec(index, "overload", short_prompt, 8, 1, 8, 2, credit));
  }
  specs.push_back(MakeSpec(index, "cancel", short_prompt, 256, 1, 1, 3, 1,
                           true));
  return specs;
}

bool ValidateRequest(const RequestSpec &spec, const RequestState &state) {
  if (!state.admitted || !state.retired || state.cancelled != spec.expected_cancelled ||
      state.commits != spec.expected_commits ||
      state.stream_messages != spec.expected_commits ||
      state.tokens.size() != static_cast<size_t>(spec.expected_commits) ||
      state.token_timestamps_ns.size() != state.tokens.size() ||
      state.final_position !=
          static_cast<int64_t>(spec.prompt.size()) + spec.expected_commits - 1 ||
      state.final_page != state.final_position / 128 ||
      state.final_checksum == 0 ||
      state.finish_reason != spec.expected_finish_reason ||
      state.first_send_ns <= 0 || state.admit_ns < state.first_send_ns ||
      state.first_commit_ns < state.admit_ns ||
      state.last_commit_ns < state.first_commit_ns ||
      state.retire_ns < state.last_commit_ns) {
    return false;
  }
  if (std::adjacent_find(state.token_timestamps_ns.begin(),
                         state.token_timestamps_ns.end(),
                         std::greater<int64_t>()) !=
      state.token_timestamps_ns.end()) {
    return false;
  }
  return spec.tag != "eos" ||
         (state.tokens.size() == 1 && state.tokens[0] == 151645);
}

void WriteIntVector(std::ofstream &output, const std::vector<int64_t> &values) {
  output << "[";
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ", ";
    output << values[index];
  }
  output << "]";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 7 && argc != 8) {
    std::cerr << "usage: persistent_admission_p4_host FUNCTION_CONFIG "
                 "GRAPH_CONFIG DEPLOY_CONFIG AIR OWNER SCENARIO [SUMMARY_JSON]"
              << std::endl;
    return 2;
  }
  const std::string function_config = argv[1];
  const std::string graph_config = argv[2];
  const std::string deploy_config = argv[3];
  const std::string air_path = argv[4];
  const int64_t owner = std::stoll(argv[5]);
  const std::string scenario = argv[6];
  const std::string summary_path = argc == 8 ? argv[7] : "";
  const char *external_weight_env =
      std::getenv("CRUISE_P4_EXTERNAL_WEIGHT_DIR");
  const std::string external_weight_dir =
      external_weight_env == nullptr ? std::string() : external_weight_env;
  const char *asset_root_env = std::getenv("CRUISE_P4_ASSET_ROOT");
  const std::string asset_root =
      asset_root_env == nullptr ? "/workspace/cruise-assets" : asset_root_env;
  const char *run_root_env = std::getenv("CRUISE_P4_RUN_ROOT");
  const std::string run_root =
      run_root_env == nullptr ? std::string() : run_root_env;
  const bool controlled_runtime_view =
      !run_root.empty() && run_root.back() != '/' &&
      external_weight_dir == run_root + "/.ge-external-view";
  const bool external_weights_allowed =
      external_weight_dir.rfind("/dev/shm/", 0) == 0 ||
      (!asset_root.empty() && asset_root.back() != '/' &&
       external_weight_dir.rfind(asset_root + "/", 0) == 0) ||
      controlled_runtime_view;
  const auto specs = BuildSpecs(scenario);
  if (access(function_config.c_str(), R_OK) != 0 ||
      access(graph_config.c_str(), R_OK) != 0 ||
      access(deploy_config.c_str(), R_OK) != 0 ||
      access(air_path.c_str(), R_OK) != 0 || owner <= 0 || specs.empty() ||
      external_weight_dir.empty() || !external_weights_allowed ||
      access(external_weight_dir.c_str(), R_OK) != 0) {
    std::cerr << "invalid P4 configuration, AIR, scenario, or owner" << std::endl;
    return 2;
  }

  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.experiment.data_flow_deploy_info_path", deploy_config.c_str()},
      {"ge.externalWeight", "1"},
      {"ge.externalWeightDir", external_weight_dir.c_str()},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
  std::cout << "P4_OWNER_START pid=" << getpid() << " owner=" << owner
            << " scenario=" << scenario << std::endl;
  auto ret = ge::GEInitialize(config);
  if (ret != ge::SUCCESS) return ret;
  auto flow_graph = BuildFlowGraph(air_path, graph_config, function_config, owner);
  const auto &graph = flow_graph.ToGeGraph();
  if (!graph.IsValid()) {
    ge::GEFinalize();
    return 1;
  }
  auto session = std::make_shared<ge::Session>(config);
  ret = session->AddGraph(kGraphId, graph);
  if (ret == ge::SUCCESS) ret = session->CompileGraph(kGraphId);
  if (ret != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return ret;
  }

  SharedState shared;
  for (const auto &spec : specs) shared.requests[spec.key].tag = spec.tag;
  uint64_t transport_seq = 0;
  int64_t event_seq = 0;
  int64_t feed_calls = 0;
  auto send = [&](const Event &event) -> bool {
    const auto status = FeedEvent(session, event, ++transport_seq);
    if (status != ge::SUCCESS) {
      std::lock_guard<std::mutex> lock(shared.mutex);
      shared.stop = true;
      shared.fetch_status = status;
      shared.changed.notify_all();
      return false;
    }
    ++feed_calls;
    return true;
  };
  auto submit = [&](size_t index, int64_t cohort_id,
                    int64_t cohort_size) -> bool {
    {
      std::lock_guard<std::mutex> lock(shared.mutex);
      auto &state = shared.requests[specs[index].key];
      ++state.send_count;
      if (state.first_send_ns == 0) state.first_send_ns = SteadyNanos();
      shared.max_pending_requests =
          std::max(shared.max_pending_requests, CountPending(shared));
    }
    return send(Admission(owner, ++event_seq, specs[index], cohort_id,
                          cohort_size));
  };
  auto send_credit = [&](size_t index, int64_t row, int64_t credit) -> bool {
    return send(Credit(owner, ++event_seq, specs[index], row, credit));
  };
  auto send_cancel = [&](size_t index, int64_t row) -> bool {
    return send(Cancel(owner, ++event_seq, specs[index], row));
  };

  std::thread drain(FetchOutputs, session, std::ref(shared));
  const int64_t started_ns = SteadyNanos();
  bool passed = true;
  auto run_closed_loop = [&](size_t begin, size_t end, size_t window) -> bool {
    size_t next = begin;
    while (next < end && next < begin + window) {
      const int64_t cohort = 1 + static_cast<int64_t>((next - begin) / window);
      if (!submit(next++, cohort, static_cast<int64_t>(window))) return false;
    }
    size_t observed = 0;
    while (observed < end - begin) {
      const size_t before = observed;
      if (!WaitFor(shared, [&](const SharedState &state) {
            return RetiredInRange(state, specs, begin, end) > before ||
                   state.rejected > 0 || state.protocol_errors > 0;
          }, 1800)) {
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(shared.mutex);
        if (shared.rejected > 0 || shared.protocol_errors > 0) return false;
        observed = RetiredInRange(shared, specs, begin, end);
      }
      while (next < end) {
        int64_t pending = 0;
        {
          std::lock_guard<std::mutex> lock(shared.mutex);
          pending = CountPending(shared);
        }
        if (pending >= static_cast<int64_t>(window)) break;
        const int64_t cohort = 1 + static_cast<int64_t>((next - begin) / window);
        if (!submit(next++, cohort, static_cast<int64_t>(window))) return false;
      }
    }
    return true;
  };
  auto run_wave = [&](size_t begin, size_t end) -> bool {
    const int64_t rejected_before = [&]() {
      std::lock_guard<std::mutex> lock(shared.mutex);
      return shared.rejected;
    }();
    const int64_t cohort = 1000 + static_cast<int64_t>(begin);
    for (size_t index = begin; index < end; ++index) {
      if (!submit(index, cohort, static_cast<int64_t>(end - begin))) return false;
    }
    return WaitFor(shared, [&](const SharedState &state) {
      return RetiredInRange(state, specs, begin, end) == end - begin ||
             state.rejected > rejected_before || state.protocol_errors > 0;
    }, 600) && [&]() {
      std::lock_guard<std::mutex> lock(shared.mutex);
      return RetiredInRange(shared, specs, begin, end) == end - begin &&
             shared.rejected == rejected_before && shared.protocol_errors == 0;
    }();
  };

  if (scenario == "primary-c4") {
    passed = run_closed_loop(0, specs.size(), kPrimaryConcurrency);
  } else {
    passed = run_closed_loop(0, 8, 4);
    passed = passed && run_wave(8, 9);
    passed = passed && run_wave(9, 13);
    if (passed) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    passed = passed && run_wave(13, 17);
    if (passed) {
      for (size_t index = 17; index < 21; ++index) {
        passed = passed && submit(index, 2000, 4);
      }
      passed = passed && WaitFor(shared, [&](const SharedState &state) {
        return AdmittedWithCommitsInRange(state, specs, 17, 21, 1) == 4;
      }, 600);
      for (size_t index = 21; index < 25; ++index) {
        passed = passed && submit(index, 0, 0);
      }
      passed = passed && WaitFor(shared, [](const SharedState &state) {
        const auto found = state.rejected_statuses.find(kStatusNoCapacity);
        return found != state.rejected_statuses.end() && found->second == 4;
      }, 120);
      for (size_t index = 17; index < 21 && passed; ++index) {
        int64_t row = -1;
        {
          std::lock_guard<std::mutex> lock(shared.mutex);
          row = shared.requests[specs[index].key].row;
        }
        passed = send_credit(index, row, 7);
      }
      passed = passed && WaitFor(shared, [&](const SharedState &state) {
        return RetiredInRange(state, specs, 17, 21) == 4;
      }, 600);
      for (size_t index = 21; index < 25 && passed; ++index) {
        passed = submit(index, 2001, 4);
      }
      passed = passed && WaitFor(shared, [&](const SharedState &state) {
        return RetiredInRange(state, specs, 21, 25) == 4;
      }, 600);
    }
    if (passed) {
      passed = submit(25, 3000, 1) && WaitFor(shared, [&](const SharedState &state) {
        return AdmittedWithCommitsInRange(state, specs, 25, 26, 1) == 1;
      }, 600);
      int64_t row = -1;
      if (passed) {
        std::lock_guard<std::mutex> lock(shared.mutex);
        row = shared.requests[specs[25].key].row;
      }
      passed = passed && send_cancel(25, row) &&
               WaitFor(shared, [&](const SharedState &state) {
                 return RetiredInRange(state, specs, 25, 26) == 1;
               }, 120);
    }
  }
  passed = passed && send(Shutdown(owner, ++event_seq)) &&
           WaitFor(shared, [](const SharedState &state) { return state.shutdown; },
                   120);
  const int64_t ended_ns = SteadyNanos();

  {
    std::lock_guard<std::mutex> lock(shared.mutex);
    int64_t expected_commits = 0;
    for (const auto &spec : specs) {
      const auto found = shared.requests.find(spec.key);
      if (found == shared.requests.end() || !ValidateRequest(spec, found->second)) {
        passed = false;
      }
      expected_commits += spec.expected_commits;
    }
    const int64_t expected_feeds = scenario == "primary-c4" ? 33 : 36;
    const int64_t minimum_calls = scenario == "primary-c4" ? 3064 : 110;
    const int64_t maximum_calls = scenario == "primary-c4" ? 3064 : 131;
    const int64_t expected_rejected = scenario == "primary-c4" ? 0 : 4;
    passed = passed && feed_calls == expected_feeds &&
             shared.aicore_calls >= minimum_calls &&
             shared.aicore_calls <= maximum_calls &&
             shared.total_commits == expected_commits &&
             shared.total_retired == static_cast<int64_t>(specs.size()) &&
             shared.single_token_outputs == expected_commits &&
             shared.max_active_rows == 4 &&
             shared.max_pending_requests ==
                 (scenario == "primary-c4" ? 4 : 8) &&
             shared.rejected == expected_rejected &&
             shared.protocol_errors == 0 && shared.shutdown;
    if (scenario == "regression") {
      passed = passed && shared.credit_acks == 4 &&
               shared.rejected_statuses[kStatusNoCapacity] == 4;
      for (size_t index = 21; index < 25; ++index) {
        passed = passed &&
                 shared.requests[specs[index].key].rejected_admissions == 1;
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(shared.mutex);
    shared.stop = true;
    shared.changed.notify_all();
  }
  drain.join();
  session->RemoveGraph(kGraphId);
  session.reset();
  ge::GEFinalize();

  std::ofstream summary(summary_path.empty() ? "/dev/null" : summary_path,
                        std::ios::trunc);
  if (summary) {
    summary << "{\n  \"gate\": \"P4-SERVICE\",\n"
            << "  \"pass\": " << (passed ? "true" : "false") << ",\n"
            << "  \"scenario\": \"" << scenario << "\",\n"
            << "  \"owner\": " << owner << ",\n"
            << "  \"feed_calls\": " << feed_calls << ",\n"
            << "  \"requests\": " << specs.size() << ",\n"
            << "  \"duration_ns\": " << ended_ns - started_ns << ",\n"
            << "  \"aicore_calls\": " << shared.aicore_calls << ",\n"
            << "  \"total_commits\": " << shared.total_commits << ",\n"
            << "  \"total_retired\": " << shared.total_retired << ",\n"
            << "  \"single_token_outputs\": " << shared.single_token_outputs << ",\n"
            << "  \"max_active_rows\": " << shared.max_active_rows << ",\n"
            << "  \"max_pending_requests\": " << shared.max_pending_requests << ",\n"
            << "  \"quiescent\": " << shared.quiescent << ",\n"
            << "  \"credit_acks\": " << shared.credit_acks << ",\n"
            << "  \"duplicate_acks\": " << shared.duplicate_acks << ",\n"
            << "  \"rejected\": " << shared.rejected << ",\n"
            << "  \"protocol_errors\": " << shared.protocol_errors << ",\n"
            << "  \"shutdown\": " << (shared.shutdown ? "true" : "false")
            << ",\n  \"rejected_statuses\": {";
    bool first_status = true;
    for (const auto &item : shared.rejected_statuses) {
      if (!first_status) summary << ", ";
      first_status = false;
      summary << "\"" << item.first << "\": " << item.second;
    }
    summary << "},\n  \"request_states\": [";
    for (size_t index = 0; index < specs.size(); ++index) {
      if (index != 0) summary << ", ";
      const auto &spec = specs[index];
      const auto &state = shared.requests[spec.key];
      summary << "{\"request\": " << spec.key.first
              << ", \"generation\": " << spec.key.second
              << ", \"tag\": \"" << spec.tag << "\""
              << ", \"row\": " << state.row
              << ", \"send_count\": " << state.send_count
              << ", \"rejected_admissions\": " << state.rejected_admissions
              << ", \"commits\": " << state.commits
              << ", \"stream_messages\": " << state.stream_messages
              << ", \"retired\": " << (state.retired ? "true" : "false")
              << ", \"cancelled\": " << (state.cancelled ? "true" : "false")
              << ", \"final_position\": " << state.final_position
              << ", \"final_page\": " << state.final_page
              << ", \"final_checksum\": " << state.final_checksum
              << ", \"finish_reason\": " << state.finish_reason
              << ", \"first_send_ns\": " << state.first_send_ns
              << ", \"admit_ns\": " << state.admit_ns
              << ", \"first_commit_ns\": " << state.first_commit_ns
              << ", \"last_commit_ns\": " << state.last_commit_ns
              << ", \"retire_ns\": " << state.retire_ns
              << ", \"tokens\": ";
      WriteIntVector(summary, state.tokens);
      summary << ", \"token_timestamps_ns\": ";
      WriteIntVector(summary, state.token_timestamps_ns);
      summary << "}";
    }
    summary << "],\n"
            << "  \"claim_boundary\": \"Quantum-Boundary Serial Admission and incremental drain only; no P5 performance or P6 concurrency claim.\"\n}\n";
  }
  std::cout << "P4_SERVICE_RESULT pass=" << (passed ? 1 : 0)
            << " scenario=" << scenario << " feed_calls=" << feed_calls
            << " aicore_calls=" << shared.aicore_calls
            << " total_commits=" << shared.total_commits
            << " total_retired=" << shared.total_retired << std::endl;
  return passed ? 0 : 20;
}
