#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
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
constexpr int32_t kBlocksPerRequest = 3;
constexpr int32_t kEventHeaderElements = 16;
constexpr int32_t kEventElements = 144;
constexpr int32_t kOutputElements = 20;
constexpr int32_t kFirstFeedTimeoutMs = 180000;
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

using RequestKey = std::pair<int64_t, int64_t>;

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
  int64_t flags = 1;
  std::array<int64_t, 4> reserved{};
  std::array<int64_t, 128> prompt{};
};

struct RequestState {
  int64_t row = -1;
  int64_t commits = 0;
  bool admitted = false;
  bool retired = false;
  bool cancelled = false;
  int64_t final_position = -1;
  int64_t final_page = -1;
  int64_t final_checksum = 0;
  int64_t finish_reason = 0;
  std::vector<int64_t> tokens;
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
  int64_t duplicate_acks = 0;
  std::map<int64_t, int64_t> rejected_statuses;
  bool shutdown = false;
  bool stop = false;
  int32_t fetch_status = ge::SUCCESS;
};

Event Admission(int64_t owner, int64_t seq, int64_t request,
                int64_t generation, const std::vector<int64_t> &prompt,
                int64_t target_steps, int64_t eos_token = 151645,
                int64_t flags = 0, int64_t credit = -1) {
  Event event;
  event.owner = owner;
  event.type = kEventAdmit;
  event.seq = seq;
  event.request = request;
  event.generation = generation;
  event.prompt_len = static_cast<int64_t>(prompt.size());
  event.target_steps = target_steps;
  event.credit = credit < 0 ? target_steps : credit;
  event.eos_token = eos_token;
  event.flags = flags;
  for (size_t index = 0; index < prompt.size() && index < event.prompt.size();
       ++index) {
    event.prompt[index] = prompt[index];
  }
  return event;
}

Event Credit(int64_t owner, int64_t seq, int64_t request,
             int64_t generation, int64_t row, int64_t credit) {
  Event event;
  event.owner = owner;
  event.type = kEventCredit;
  event.seq = seq;
  event.request = request;
  event.generation = generation;
  event.row = row;
  event.credit = credit;
  event.flags = 0;
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
  event.flags = 0;
  return event;
}

Event Shutdown(int64_t owner, int64_t seq) {
  Event event;
  event.owner = owner;
  event.type = kEventShutdown;
  event.seq = seq;
  event.row = -1;
  event.flags = 0;
  return event;
}

ge::dflow::FlowGraph BuildFlowGraph(const std::string &air_path,
                                    const std::string &graph_config,
                                    const std::string &function_config,
                                    int64_t owner) {
  using namespace ge::dflow;
  auto event = FlowData("Event", 0);
  auto decoder = GraphPp("persistent_decoder_p3_graph_pp", [air_path]() {
    ge::Graph graph("PersistentDecoderP3AIR");
    const auto status = graph.LoadFromFile(air_path.c_str());
    std::cout << "P3_AIR_LOAD status=" << status
              << " valid=" << graph.IsValid() << std::endl;
    return graph;
  });
  decoder.SetCompileConfig(graph_config.c_str());
  auto controller = FunctionPp("persistent_decoder_p3_pp")
                        .SetCompileConfig(function_config.c_str())
                        .SetInitParam("owner_instance_id", owner);
  controller.AddInvokedClosure("decode_graph_0", decoder);
  auto node = FlowNode("persistent_decoder_p3_node", 1, 1);
  node.AddPp(controller).SetInput(0, event);
  FlowGraph graph("cruise_persistent_decoder_p3");
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
          startup_failures < kStartupRetryLimit) {
        ++startup_failures;
      } else {
        shared.fetch_status = ret;
        shared.changed.notify_all();
        return;
      }
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
    {
      std::lock_guard<std::mutex> lock(shared.mutex);
      ++shared.fetch_calls;
      const RequestKey key{values[3], values[4]};
      auto &request = shared.requests[key];
      if (values[1] == kOutputAdmitAck) {
        request.row = values[5];
        request.admitted = true;
      } else if (values[1] == kOutputCommit) {
        request.row = values[5];
        request.commits = values[6];
        request.tokens.push_back(values[7]);
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
      } else if (values[1] == kOutputQuiescent) {
        ++shared.quiescent;
      } else if (values[1] == kOutputRejected) {
        ++shared.rejected;
        ++shared.rejected_statuses[values[13]];
      } else if (values[1] == kOutputCumulativeAck) {
        ++shared.duplicate_acks;
      } else if (values[1] == kOutputShutdown) {
        shared.aicore_calls = values[14];
        shared.total_commits = values[15];
        shared.total_retired = values[16];
        shared.shutdown = true;
      }
      std::cout << "P3_OUTPUT owner=" << values[0]
                << " type=" << values[1]
                << " event_seq=" << values[2]
                << " request=" << values[3]
                << " generation=" << values[4]
                << " row=" << values[5]
                << " commit_seq=" << values[6]
                << " token=" << values[7]
                << " position=" << values[8]
                << " page=" << values[9]
                << " remaining=" << values[10]
                << " credit=" << values[11]
                << " cancel_seq=" << values[12]
                << " status=" << values[13]
                << " aicore_calls=" << values[14]
                << " total_commits=" << values[15]
                << " total_retired=" << values[16]
                << " kv_checksum=" << values[17]
                << " finish_reason=" << values[18]
                << " quiescent=" << values[19]
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

ge::Status FeedEvent(const std::shared_ptr<ge::Session> &session,
                     const Event &event, uint64_t transport_seq,
                     int32_t timeout_ms) {
  ge::DataFlowInfo flow_info;
  flow_info.SetTransactionId(transport_seq);
  return session->FeedDataFlowGraph(kGraphId, {MakeEvent(event)}, flow_info,
                                    timeout_ms);
}

std::vector<std::vector<int64_t>> MakePrompts(const std::string &scenario) {
  if (scenario == "b1-short") return {{9707, 9708, 9709, 9710}};
  if (scenario == "b1-eos") {
    return {{151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553,
             54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645,
             198, 151644, 872, 198, 5598, 458, 4287, 2033, 13, 3155,
             537, 2550, 894, 1467, 13, 151645, 198, 151644, 77091, 198}};
  }
  if (scenario == "lifecycle") {
    return {{9707, 9708, 9709, 9710}};
  }
  if (scenario == "b1-full" || scenario == "b1-backpressure") {
    std::vector<int64_t> prompt(128);
    for (int64_t index = 0; index < 128; ++index) prompt[index] = 1000 + index;
    return {prompt};
  }
  if (scenario == "b4-mixed") {
    std::vector<std::vector<int64_t>> prompts;
    for (int64_t length : {1LL, 7LL, 32LL, 128LL}) {
      std::vector<int64_t> prompt(static_cast<size_t>(length));
      for (int64_t index = 0; index < length; ++index) {
        prompt[static_cast<size_t>(index)] = 2000 + length + index;
      }
      prompts.push_back(std::move(prompt));
    }
    return prompts;
  }
  return {};
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 7 && argc != 8) {
    std::cerr << "usage: persistent_decoder_p3_host FUNCTION_CONFIG GRAPH_CONFIG "
                 "DEPLOY_CONFIG AIR OWNER SCENARIO [SUMMARY_JSON]" << std::endl;
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
      std::getenv("CRUISE_P3_EXTERNAL_WEIGHT_DIR");
  const std::string external_weight_dir =
      external_weight_env == nullptr ? std::string() : external_weight_env;
  const char *asset_root_env = std::getenv("CRUISE_P3_ASSET_ROOT");
  const std::string asset_root =
      asset_root_env == nullptr ? "/workspace/cruise-assets" : asset_root_env;
  const char *run_root_env = std::getenv("CRUISE_P3_RUN_ROOT");
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
  if (access(function_config.c_str(), R_OK) != 0 ||
      access(graph_config.c_str(), R_OK) != 0 ||
      access(deploy_config.c_str(), R_OK) != 0 || access(air_path.c_str(), R_OK) != 0 ||
      owner <= 0 || external_weight_dir.empty() ||
      !external_weights_allowed ||
      access(external_weight_dir.c_str(), R_OK) != 0) {
    std::cerr << "invalid P3 configuration, AIR, or owner" << std::endl;
    return 2;
  }
  const auto prompts = MakePrompts(scenario);
  if (prompts.empty()) {
    std::cerr << "unknown scenario: " << scenario << std::endl;
    return 2;
  }
  const std::array<int64_t, 4> targets =
      (scenario == "b1-short" || scenario == "b1-eos")
          ? std::array<int64_t, 4>{8, 0, 0, 0}
      : scenario == "lifecycle" ? std::array<int64_t, 4>{256, 0, 0, 0}
      : (scenario == "b1-full" || scenario == "b1-backpressure")
          ? std::array<int64_t, 4>{256, 0, 0, 0}
                               : std::array<int64_t, 4>{8, 16, 64, 256};

  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.experiment.data_flow_deploy_info_path", deploy_config.c_str()},
      {"ge.externalWeight", "1"},
      {"ge.externalWeightDir", external_weight_dir.c_str()},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
  std::cout << "P3_OWNER_START pid=" << getpid() << " owner=" << owner
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
  uint64_t transport_seq = 0;
  int64_t feed_calls = 0;
  bool first_feed_attempted = false;
  auto send = [&](const Event &event) -> bool {
    const bool first_feed = !first_feed_attempted;
    first_feed_attempted = true;
    const int32_t timeout_ms =
        first_feed ? kFirstFeedTimeoutMs : kFeedTimeoutMs;
    const auto started = std::chrono::steady_clock::now();
    const auto status =
        FeedEvent(session, event, ++transport_seq, timeout_ms);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    if (status != ge::SUCCESS) {
      std::cerr << "P3_FEED_FAIL event_seq=" << event.seq
                << " type=" << event.type << " status=" << status
                << " first_feed=" << first_feed
                << " timeout_ms=" << timeout_ms
                << " elapsed_ms=" << elapsed_ms << std::endl;
      std::lock_guard<std::mutex> lock(shared.mutex);
      shared.stop = true;
      shared.fetch_status = status;
      shared.changed.notify_all();
      return false;
    }
    ++feed_calls;
    std::cout << "P3_FEED_OK event_seq=" << event.seq
              << " type=" << event.type
              << " first_feed=" << first_feed
              << " timeout_ms=" << timeout_ms
              << " elapsed_ms=" << elapsed_ms << std::endl;
    return true;
  };
  std::thread drain;
  if (scenario == "b1-backpressure") {
    drain = std::thread([&session, &shared]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(1500));
      FetchOutputs(session, shared);
    });
  } else {
    drain = std::thread(FetchOutputs, session, std::ref(shared));
  }
  bool passed = true;
  std::vector<RequestKey> keys;
  if (scenario == "lifecycle") {
    const auto prompt = prompts.front();
    const RequestKey first{11000, 10};
    keys.push_back(first);
    const auto first_admit = Admission(owner, 1, first.first, first.second,
                                       prompt, 256);
    passed = passed && send(first_admit) &&
             WaitFor(shared, [&](const SharedState &state) {
               const auto found = state.requests.find(first);
               return found != state.requests.end() && found->second.admitted;
             }, 60);
    passed = passed && send(first_admit) &&
             WaitFor(shared, [](const SharedState &state) {
               return state.duplicate_acks >= 1;
             }, 30);
    const auto first_cancel = Cancel(owner, 2, first.first, first.second, 0, 1);
    passed = passed && send(first_cancel) &&
             WaitFor(shared, [&](const SharedState &state) {
               const auto found = state.requests.find(first);
               return found != state.requests.end() && found->second.retired &&
                      found->second.cancelled;
             }, 60);
    passed = passed && send(first_cancel) &&
             WaitFor(shared, [](const SharedState &state) {
               return state.duplicate_acks >= 2;
             }, 30);
    passed = passed && send(first_admit) &&
             WaitFor(shared, [](const SharedState &state) {
               return state.rejected_statuses.count(2) != 0;
             }, 30);

    const RequestKey second{11001, 11};
    keys.push_back(second);
    passed = passed && send(Admission(owner, 3, second.first, second.second,
                                      prompt, 256, 151645, 1, 1)) &&
             WaitFor(shared, [&](const SharedState &state) {
               const auto found = state.requests.find(second);
               return found != state.requests.end() && found->second.admitted &&
                      found->second.commits >= 1;
             }, 120);
    passed = passed && send(Credit(owner, 4, second.first, 10, 0, 1)) &&
             WaitFor(shared, [](const SharedState &state) {
               return state.rejected_statuses.count(4) != 0;
             }, 30);
    passed = passed && send(Credit(owner, 5, second.first, second.second, 0, 1)) &&
             WaitFor(shared, [&](const SharedState &state) {
               const auto found = state.requests.find(second);
               return found != state.requests.end() && found->second.commits >= 2;
             }, 120);
    passed = passed && send(Cancel(owner, 6, second.first, second.second, 0, 1)) &&
             WaitFor(shared, [&](const SharedState &state) {
               const auto found = state.requests.find(second);
               return found != state.requests.end() && found->second.retired &&
                      found->second.cancelled;
             }, 60);

    const RequestKey reused{11002, 12};
    keys.push_back(reused);
    passed = passed && send(Admission(owner, 7, reused.first, reused.second,
                                      prompt, 8)) &&
             WaitFor(shared, [&](const SharedState &state) {
               const auto found = state.requests.find(reused);
               return found != state.requests.end() && found->second.admitted &&
                      found->second.retired;
             }, 120);
    passed = passed && send(Cancel(owner, 5, second.first, second.second, 0, 1)) &&
             WaitFor(shared, [](const SharedState &state) {
               const auto found = state.rejected_statuses.find(2);
               return found != state.rejected_statuses.end() && found->second >= 2;
             }, 30);
    passed = passed && send(Shutdown(owner, 8)) &&
             WaitFor(shared, [](const SharedState &state) { return state.shutdown; },
                     120);
  } else {
    for (size_t index = 0; index < prompts.size(); ++index) {
      const RequestKey key{10000 + static_cast<int64_t>(index),
                           1 + static_cast<int64_t>(index)};
      keys.push_back(key);
      const int64_t eos_token = 151645;
      const int64_t flags =
          scenario == "b1-full" || scenario == "b1-backpressure" ? 1 : 0;
      passed = passed && send(Admission(owner, static_cast<int64_t>(index + 1),
                                        key.first, key.second, prompts[index],
                                        targets[index], eos_token, flags));
    }
    passed = passed && WaitFor(shared, [&](const SharedState &state) {
      if (state.rejected > 0) return true;
      for (const auto &key : keys) {
        const auto found = state.requests.find(key);
        if (found == state.requests.end() || !found->second.admitted ||
            !found->second.retired) return false;
      }
      return true;
    }, scenario == "b1-full" || scenario == "b1-backpressure" ? 1800 : 600);
    const int64_t shutdown_seq = static_cast<int64_t>(prompts.size() + 1);
    passed = passed && send(Shutdown(owner, shutdown_seq)) &&
             WaitFor(shared, [](const SharedState &state) { return state.shutdown; },
                     120);
  }
  {
    std::lock_guard<std::mutex> lock(shared.mutex);
    if (scenario == "lifecycle") {
      for (size_t index = 0; index < keys.size(); ++index) {
        const auto found = shared.requests.find(keys[index]);
        if (found == shared.requests.end() || found->second.row != 0 ||
            !found->second.retired) {
          passed = false;
          continue;
        }
        if (index < 2) passed = passed && found->second.cancelled;
      }
      const auto second_it = shared.requests.find(keys[1]);
      const auto reused_it = shared.requests.find(keys[2]);
      if (second_it == shared.requests.end() || reused_it == shared.requests.end()) {
        passed = false;
      } else {
        const auto &second = second_it->second;
        const auto &reused = reused_it->second;
        const auto stale_it = shared.rejected_statuses.find(2);
        const auto generation_it = shared.rejected_statuses.find(4);
        passed = passed && second.commits == 2 && reused.commits == 8 &&
                 !reused.cancelled && reused.final_position == 11 &&
                 reused.final_page == 0 && reused.final_checksum != 0 &&
                 shared.total_retired == 3 && shared.duplicate_acks == 2 &&
                 stale_it != shared.rejected_statuses.end() && stale_it->second == 2 &&
                 generation_it != shared.rejected_statuses.end() &&
                 generation_it->second == 1 && feed_calls == 12 && shared.shutdown;
      }
    } else {
      for (size_t index = 0; index < keys.size(); ++index) {
        const auto found = shared.requests.find(keys[index]);
        if (found == shared.requests.end()) {
          passed = false;
          continue;
        }
        const auto &state = found->second;
        const int64_t expected_commits = scenario == "b1-eos" ? 1 : targets[index];
        const int64_t expected_position =
            static_cast<int64_t>(prompts[index].size()) + expected_commits - 1;
        passed = passed && state.commits == expected_commits &&
                 state.final_position == expected_position &&
                 state.final_checksum != 0 &&
                 state.final_page == expected_position / 128;
        if (scenario == "b1-eos") {
          passed = passed && state.finish_reason == 1;
        }
      }
      passed = passed && feed_calls == static_cast<int64_t>(prompts.size() + 1) &&
               shared.rejected == 0 && shared.shutdown;
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
    summary << "{\n  \"gate\": \"P3-OWNER\",\n"
            << "  \"pass\": " << (passed ? "true" : "false") << ",\n"
            << "  \"scenario\": \"" << scenario << "\",\n"
            << "  \"owner\": " << owner << ",\n"
            << "  \"feed_calls\": " << feed_calls << ",\n"
            << "  \"requests\": " << keys.size() << ",\n"
            << "  \"aicore_calls\": " << shared.aicore_calls << ",\n"
            << "  \"total_commits\": " << shared.total_commits << ",\n"
            << "  \"total_retired\": " << shared.total_retired << ",\n"
            << "  \"quiescent\": " << shared.quiescent << ",\n"
            << "  \"duplicate_acks\": " << shared.duplicate_acks << ",\n"
            << "  \"rejected_statuses\": {";
    bool first_status = true;
    for (const auto &item : shared.rejected_statuses) {
      if (!first_status) summary << ", ";
      first_status = false;
      summary << "\"" << item.first << "\": " << item.second;
    }
    summary << "},\n  \"request_states\": [";
    bool first_request = true;
    for (const auto &key : keys) {
      const auto found = shared.requests.find(key);
      if (found == shared.requests.end()) continue;
      if (!first_request) summary << ", ";
      first_request = false;
      const auto &state = found->second;
      summary << "{\"request\": " << key.first
              << ", \"generation\": " << key.second
              << ", \"row\": " << state.row
              << ", \"commits\": " << state.commits
              << ", \"retired\": " << (state.retired ? "true" : "false")
              << ", \"cancelled\": " << (state.cancelled ? "true" : "false")
              << ", \"final_position\": " << state.final_position
              << ", \"final_page\": " << state.final_page
              << ", \"final_checksum\": " << state.final_checksum
              << ", \"finish_reason\": " << state.finish_reason
              << ", \"tokens\": [";
      for (size_t index = 0; index < state.tokens.size(); ++index) {
        if (index != 0) summary << ", ";
        summary << state.tokens[index];
      }
      summary << "]}";
    }
    summary << "],\n"
            << "  \"claim_boundary\": \"Owner AIR invocation and bounded prompt/decode correctness; no Graph oracle or performance claim.\"\n}\n";
  }
  std::cout << "P3_OWNER_RESULT pass=" << (passed ? 1 : 0)
            << " scenario=" << scenario << " feed_calls=" << feed_calls
            << " aicore_calls=" << shared.aicore_calls
            << " total_commits=" << shared.total_commits
            << " total_retired=" << shared.total_retired << std::endl;
  return passed ? 0 : 20;
}
