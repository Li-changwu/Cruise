#define main cruise_p4_harness_main
#include "../persistent_admission_p4/persistent_admission_p4_host.cpp"
#undef main

#include <sstream>

namespace {

bool ReadInteger(std::istringstream &input, int64_t &value) {
  input >> value;
  return !input.fail();
}

bool ReadAdmission(std::istringstream &input, int64_t owner, int64_t seq,
                   Event &event) {
  event.owner = owner;
  event.type = kEventAdmit;
  event.seq = seq;
  if (!ReadInteger(input, event.request) ||
      !ReadInteger(input, event.generation) ||
      !ReadInteger(input, event.target_steps) ||
      !ReadInteger(input, event.credit) ||
      !ReadInteger(input, event.eos_token) ||
      !ReadInteger(input, event.flags) ||
      !ReadInteger(input, event.reserved[0]) ||
      !ReadInteger(input, event.reserved[1]) ||
      !ReadInteger(input, event.prompt_len) || event.prompt_len <= 0 ||
      event.prompt_len > static_cast<int64_t>(event.prompt.size())) {
    return false;
  }
  for (int64_t index = 0; index < event.prompt_len; ++index) {
    if (!ReadInteger(input, event.prompt[static_cast<size_t>(index)])) return false;
  }
  std::string extra;
  return !(input >> extra);
}

bool ReadRowEvent(std::istringstream &input, int64_t owner, int64_t type,
                  int64_t seq, Event &event) {
  event.owner = owner;
  event.type = type;
  event.seq = seq;
  if (!ReadInteger(input, event.request) ||
      !ReadInteger(input, event.generation) || !ReadInteger(input, event.row)) {
    return false;
  }
  if (type == kEventCredit) {
    if (!ReadInteger(input, event.credit)) return false;
  } else if (!ReadInteger(input, event.cancel_seq)) {
    return false;
  }
  std::string extra;
  return !(input >> extra);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: persistent_owner_transport FUNCTION_CONFIG "
                 "GRAPH_CONFIG DEPLOY_CONFIG AIR OWNER"
              << std::endl;
    return 2;
  }
  const std::string function_config = argv[1];
  const std::string graph_config = argv[2];
  const std::string deploy_config = argv[3];
  const std::string air_path = argv[4];
  const int64_t owner = std::stoll(argv[5]);
  const char *external_weight_env =
      std::getenv("CRUISE_P5_EXTERNAL_WEIGHT_DIR");
  const std::string external_weight_dir =
      external_weight_env == nullptr ? std::string() : external_weight_env;
  const char *asset_root_env = std::getenv("CRUISE_P5_ASSET_ROOT");
  const std::string asset_root =
      asset_root_env == nullptr ? "/workspace/cruise-assets" : asset_root_env;
  const char *run_root_env = std::getenv("CRUISE_P5_RUN_ROOT");
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
      access(deploy_config.c_str(), R_OK) != 0 ||
      access(air_path.c_str(), R_OK) != 0 || owner <= 0 ||
      external_weight_dir.empty() || !external_weights_allowed ||
      access(external_weight_dir.c_str(), R_OK) != 0) {
    std::cerr << "invalid P5 Owner configuration" << std::endl;
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
  std::cout << "P5_OWNER_PHASE phase=ge_initialize_start" << std::endl;
  auto ret = ge::GEInitialize(config);
  std::cout << "P5_OWNER_PHASE phase=ge_initialize_returned status=" << ret
            << std::endl;
  if (ret != ge::SUCCESS) return ret;
  std::cout << "P5_OWNER_PHASE phase=graph_build_start" << std::endl;
  auto flow_graph = BuildFlowGraph(air_path, graph_config, function_config, owner);
  const auto &graph = flow_graph.ToGeGraph();
  std::cout << "P5_OWNER_PHASE phase=graph_build_returned valid="
            << (graph.IsValid() ? 1 : 0) << std::endl;
  if (!graph.IsValid()) {
    ge::GEFinalize();
    return 1;
  }
  auto session = std::make_shared<ge::Session>(config);
  ret = session->AddGraph(kGraphId, graph);
  std::cout << "P5_OWNER_PHASE phase=add_graph_returned status=" << ret
            << std::endl;
  if (ret == ge::SUCCESS) {
    std::cout << "P5_OWNER_PHASE phase=compile_start" << std::endl;
    ret = session->CompileGraph(kGraphId);
    std::cout << "P5_OWNER_PHASE phase=compile_returned status=" << ret
              << std::endl;
  }
  if (ret != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return ret;
  }

  SharedState shared;
  uint64_t transport_seq = 0;
  int64_t event_seq = 0;
  int64_t feed_calls = 0;
  std::cout << "P5_OWNER_PHASE phase=drain_start" << std::endl;
  std::thread drain(FetchOutputs, session, std::ref(shared));
  std::cout << "P5_OWNER_PHASE phase=drain_started" << std::endl;
  std::cout << "P5_OWNER_READY pid=" << getpid() << " owner=" << owner
            << " route=persistent_device_model_owner" << std::endl;

  bool command_ok = true;
  bool shutdown_sent = false;
  std::string line;
  while (command_ok && std::getline(std::cin, line)) {
    std::istringstream input(line);
    std::string command;
    input >> command;
    Event event;
    if (command == "ADMIT") {
      command_ok = ReadAdmission(input, owner, ++event_seq, event);
      if (command_ok) {
        std::lock_guard<std::mutex> lock(shared.mutex);
        auto &state = shared.requests[{event.request, event.generation}];
        ++state.send_count;
        if (state.first_send_ns == 0) state.first_send_ns = SteadyNanos();
      }
    } else if (command == "CREDIT") {
      command_ok = ReadRowEvent(input, owner, kEventCredit, ++event_seq, event);
    } else if (command == "CANCEL") {
      command_ok = ReadRowEvent(input, owner, kEventCancel, ++event_seq, event);
    } else if (command == "SHUTDOWN") {
      std::string extra;
      command_ok = !(input >> extra);
      event = Shutdown(owner, ++event_seq);
      shutdown_sent = command_ok;
    } else {
      command_ok = false;
    }
    if (!command_ok) {
      std::cerr << "P5_OWNER_PROTOCOL_ERROR line=" << line << std::endl;
      break;
    }
    ret = FeedEvent(session, event, ++transport_seq);
    if (ret != ge::SUCCESS) {
      command_ok = false;
      break;
    }
    ++feed_calls;
    if (shutdown_sent) break;
  }

  bool clean_shutdown = false;
  if (shutdown_sent) {
    clean_shutdown = WaitFor(
        shared, [](const SharedState &state) { return state.shutdown; }, 120);
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
  std::cout << "P5_OWNER_EXIT clean=" << (clean_shutdown ? 1 : 0)
            << " feed_calls=" << feed_calls
            << " aicore_calls=" << shared.aicore_calls
            << " total_commits=" << shared.total_commits
            << " total_retired=" << shared.total_retired << std::endl;
  return command_ok && clean_shutdown ? 0 : 20;
}
