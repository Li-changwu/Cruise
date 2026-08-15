#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <acl/acl_prof.h>

#ifdef CRUISE_RESIDENT_DEVICE_TRANSFER_PLUGIN
#include <dlfcn.h>
#endif

#include "resident_epoch_bridge.h"
#include "resident_epoch_protocol.h"

namespace {
constexpr uint32_t kRequestMagic = 0x71317131U;
constexpr uint32_t kResponseMagic = 0x71327132U;
constexpr uint16_t kProtocolVersion = CRUISE_SIDECAR_PROTOCOL_VERSION;
constexpr uint16_t kExecute = 1;
constexpr uint16_t kWarmUp = 2;
constexpr uint16_t kShutdown = 3;
constexpr uint16_t kDeviceIpcExecute = 5;
constexpr uint16_t kStartProfiling = 6;
constexpr uint16_t kStopProfiling = 7;
constexpr int32_t kBatchSize = 4;
constexpr int32_t kMaxEpochSteps = 8;

#pragma pack(push, 1)
struct Request {
  uint32_t magic;
  uint16_t version;
  uint16_t operation;
  int32_t request_count;
  int32_t max_steps;
  uint64_t transfer_id;
  int64_t token_ids[kBatchSize];
  int64_t positions[kBatchSize];
  int32_t sequence_lengths[kBatchSize];
  int32_t eos_token_ids[kBatchSize];
  int32_t row_generations[kBatchSize];
};

struct Response {
  uint32_t magic;
  int32_t transport_status;
  int32_t device_status;
  int32_t model_calls;
  int32_t feed_calls;
  int32_t fetch_calls;
  int32_t commit_state;
  int32_t reserved;
  int64_t wall_us;
  int64_t native_cpu_us;
  int64_t device_kv_transfer_wall_us;
  int64_t device_kv_transfer_cpu_us;
  int64_t declared_input_bytes;
  int64_t declared_output_bytes;
  int32_t executed[kBatchSize];
  int32_t row_generations[kBatchSize];
  int64_t token_ids[kBatchSize * kMaxEpochSteps];
};
#pragma pack(pop)

static_assert(sizeof(Request) == CRUISE_SIDECAR_REQUEST_BYTES,
              "resident epoch request ABI changed");
static_assert(sizeof(Response) == CRUISE_SIDECAR_RESPONSE_BYTES,
              "resident epoch response ABI changed");
static_assert(sizeof(ResidentEpochIpcMetadata) ==
                  CRUISE_RESIDENT_IPC_METADATA_BYTES,
              "resident Device IPC metadata ABI changed");

bool RebindDynamicProfiling() {
  const char *enabled =
      std::getenv("VLLM_ASCEND_RESIDENT_EPOCH_DYNAMIC_PROFILING");
  const char *mode = std::getenv("PROFILING_MODE");
  if ((enabled == nullptr || std::strcmp(enabled, "1") != 0) &&
      (mode == nullptr || std::strcmp(mode, "dynamic") != 0)) {
    return true;
  }

  char pid_text[32] = {};
  std::snprintf(pid_text, sizeof(pid_text), "%ld", static_cast<long>(getpid()));
  if (setenv("PROFILING_MODE", "dynamic", 1) != 0 ||
      setenv("DYNAMIC_PROFILING_KEY_PID", pid_text, 1) != 0) {
    return false;
  }

  const char *temporary_directory = std::getenv("TMPDIR");
  if (temporary_directory == nullptr || temporary_directory[0] == '\0') {
    return true;
  }
  std::string binding_path =
      std::string(temporary_directory) + "/dynamic-profiling-binding-" +
      pid_text + ".tsv";
  std::FILE *binding = std::fopen(binding_path.c_str(), "w");
  if (binding == nullptr) return false;
  const int written = std::fprintf(
      binding,
      "key\tvalue\nprofiling_mode\tdynamic\ndynamic_profiling_key_pid\t%s\n",
      pid_text);
  return std::fclose(binding) == 0 && written > 0;
}

struct PostWarmupProfiler {
  aclprofConfig *config = nullptr;
  bool initialized = false;
};

int32_t StartPostWarmupProfiler(PostWarmupProfiler *profiler) {
  if (profiler == nullptr || profiler->initialized || profiler->config != nullptr) {
    return 75;
  }
  const char *output =
      std::getenv("VLLM_ASCEND_RESIDENT_EPOCH_PROFILE_OUTPUT");
  if (output == nullptr || std::strncmp(output, "/dev/shm/", 9) != 0) {
    return 76;
  }
  const aclError init_status = aclprofInit(output, std::strlen(output));
  if (init_status != ACL_SUCCESS) {
    std::fprintf(stderr, "resident profiler init failed: %d\n", init_status);
    return 77;
  }
  profiler->initialized = true;
  uint32_t device_id = 0;
  profiler->config = aclprofCreateConfig(
      &device_id, 1, ACL_AICORE_NONE, nullptr, ACL_PROF_TASK_TIME_L0);
  if (profiler->config == nullptr) {
    aclprofFinalize();
    profiler->initialized = false;
    return 78;
  }
  const aclError start_status = aclprofStart(profiler->config);
  if (start_status != ACL_SUCCESS) {
    std::fprintf(stderr, "resident profiler start failed: %d\n", start_status);
    aclprofDestroyConfig(profiler->config);
    profiler->config = nullptr;
    aclprofFinalize();
    profiler->initialized = false;
    return 79;
  }
  return 0;
}

int32_t StopPostWarmupProfiler(PostWarmupProfiler *profiler) {
  if (profiler == nullptr || !profiler->initialized || profiler->config == nullptr) {
    return 80;
  }
  const aclError stop_status = aclprofStop(profiler->config);
  const aclError destroy_status = aclprofDestroyConfig(profiler->config);
  profiler->config = nullptr;
  const aclError finalize_status = aclprofFinalize();
  profiler->initialized = false;
  if (stop_status != ACL_SUCCESS || destroy_status != ACL_SUCCESS ||
      finalize_status != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "resident profiler stop failed: stop=%d destroy=%d finalize=%d\n",
                 stop_status, destroy_status, finalize_status);
    return 81;
  }
  return 0;
}

bool ReadAll(int fd, void *buffer, size_t bytes) {
  auto *cursor = static_cast<uint8_t *>(buffer);
  while (bytes > 0) {
    const ssize_t count = read(fd, cursor, bytes);
    if (count == 0) return false;
    if (count < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
  return true;
}

bool WriteAll(int fd, const void *buffer, size_t bytes) {
  const auto *cursor = static_cast<const uint8_t *>(buffer);
  while (bytes > 0) {
    const ssize_t count = write(fd, cursor, bytes);
    if (count < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
  return true;
}

Response EmptyResponse(int32_t status) {
  Response response{};
  response.magic = kResponseMagic;
  response.transport_status = status;
  response.device_status = -1;
  response.commit_state = CRUISE_EPOCH_PREPARED;
  for (int32_t &executed : response.executed) executed = 0;
  for (int32_t &generation : response.row_generations) generation = 0;
  for (int64_t &token : response.token_ids) token = -1;
  return response;
}

int CreateListener(const char *path) {
  if (path == nullptr || std::strncmp(path, "/dev/shm/", 9) != 0) {
    return -1;
  }
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_un address{};
  if (std::strlen(path) >= sizeof(address.sun_path)) {
    close(fd);
    return -1;
  }
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, path, sizeof(address.sun_path) - 1);
  unlink(path);
  if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
      listen(fd, 1) != 0) {
    close(fd);
    unlink(path);
    return -1;
  }
  return fd;
}

#ifdef CRUISE_RESIDENT_DEVICE_TRANSFER_PLUGIN
struct DeviceTransferPlugin {
  void *handle = nullptr;
  ResidentDeviceTransferPrepare prepare = nullptr;
  ResidentDeviceTransferDestroy destroy = nullptr;
};

bool LoadDeviceTransferPlugin(const char *server_path,
                              DeviceTransferPlugin *plugin) {
  if (server_path == nullptr || plugin == nullptr) return false;
  const std::string executable(server_path);
  const auto separator = executable.find_last_of('/');
  if (separator == std::string::npos) return false;
  const std::string library_path =
      executable.substr(0, separator + 1) + "libresident_device_transfer.so";
  void *handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) return false;
  auto prepare = reinterpret_cast<ResidentDeviceTransferPrepare>(
      dlsym(handle, "resident_device_transfer_prepare"));
  auto destroy = reinterpret_cast<ResidentDeviceTransferDestroy>(
      dlsym(handle, "resident_device_transfer_destroy"));
  if (prepare == nullptr || destroy == nullptr) {
    dlclose(handle);
    return false;
  }
  plugin->handle = handle;
  plugin->prepare = prepare;
  plugin->destroy = destroy;
  return true;
}

void CloseDeviceTransferPlugin(DeviceTransferPlugin *plugin) {
  if (plugin == nullptr || plugin->handle == nullptr) return;
  dlclose(plugin->handle);
  plugin->handle = nullptr;
  plugin->prepare = nullptr;
  plugin->destroy = nullptr;
}
#endif
}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr,
                 "usage: %s SOCKET AIR GRAPH_CONFIG FUNC_CONFIG WEIGHTS TILING\n",
                 argv[0]);
    return 64;
  }
  if (!RebindDynamicProfiling()) return 74;
  const char *socket_path = argv[1];
  const int listener = CreateListener(socket_path);
  if (listener < 0) return 65;
  const int client = accept(listener, nullptr, nullptr);
  if (client < 0) {
    close(listener);
    unlink(socket_path);
    return 66;
  }

  int32_t create_status = -1;
  void *engine = resident_epoch_create(argv[2], argv[3], argv[4], argv[5],
                                       argv[6], &create_status);
#ifdef CRUISE_RESIDENT_DEVICE_TRANSFER_PLUGIN
  DeviceTransferPlugin transfer_plugin;
  if (engine != nullptr &&
      !LoadDeviceTransferPlugin(argv[0], &transfer_plugin)) {
    resident_epoch_destroy(engine);
    engine = nullptr;
    create_status = 8;
  }
  if (engine != nullptr && resident_epoch_install_device_transfer(
                             transfer_plugin.prepare,
                             transfer_plugin.destroy) != 0) {
    resident_epoch_destroy(engine);
    CloseDeviceTransferPlugin(&transfer_plugin);
    engine = nullptr;
    create_status = 9;
  }
#endif
  Response startup = EmptyResponse(engine == nullptr ? 100 + create_status : 0);
  if (!WriteAll(client, &startup, sizeof(startup)) || engine == nullptr) {
    if (engine != nullptr) resident_epoch_destroy(engine);
#ifdef CRUISE_RESIDENT_DEVICE_TRANSFER_PLUGIN
    CloseDeviceTransferPlugin(&transfer_plugin);
#endif
    close(client);
    close(listener);
    unlink(socket_path);
    return engine == nullptr ? 100 + create_status : 67;
  }

  int exit_status = 0;
  PostWarmupProfiler profiler;
  while (true) {
    Request request{};
    if (!ReadAll(client, &request, sizeof(request))) {
      exit_status = 68;
      break;
    }
    Response response = EmptyResponse(0);
    const bool direct_device_import = request.operation == kDeviceIpcExecute;
    std::unique_ptr<ResidentEpochIpcMetadata> ipc_metadata;
    if (request.magic != kRequestMagic ||
        request.version != kProtocolVersion) {
      response.transport_status = 69;
    } else if (request.operation == kShutdown) {
      if (profiler.initialized) StopPostWarmupProfiler(&profiler);
      if (!WriteAll(client, &response, sizeof(response))) exit_status = 70;
      break;
    } else if (request.operation == kStartProfiling) {
      response.transport_status = StartPostWarmupProfiler(&profiler);
      response.device_status = response.transport_status == 0 ? 0 : -1;
      response.commit_state = response.transport_status == 0
                                  ? CRUISE_EPOCH_COMMITTED
                                  : CRUISE_EPOCH_PREPARED;
    } else if (request.operation == kStopProfiling) {
      response.transport_status = StopPostWarmupProfiler(&profiler);
      response.device_status = response.transport_status == 0 ? 0 : -1;
      response.commit_state = response.transport_status == 0
                                  ? CRUISE_EPOCH_COMMITTED
                                  : CRUISE_EPOCH_EXECUTING;
    } else if (request.operation != kExecute &&
               request.operation != kWarmUp &&
               request.operation != kDeviceIpcExecute) {
      response.transport_status = 71;
    } else {
      if (direct_device_import) {
        ipc_metadata.reset(new ResidentEpochIpcMetadata());
        if (!ReadAll(client, ipc_metadata.get(), sizeof(*ipc_metadata))) {
          exit_status = 73;
          break;
        }
      }
      response.transport_status = resident_epoch_execute(
          engine, request.request_count, request.max_steps,
          request.token_ids, request.positions, request.sequence_lengths,
          request.eos_token_ids, request.row_generations,
          response.token_ids, response.executed, response.row_generations,
          &response.model_calls, &response.device_status,
          &response.feed_calls, &response.fetch_calls,
          &response.commit_state, &response.reserved, &response.wall_us,
          &response.native_cpu_us, &response.device_kv_transfer_wall_us,
          &response.device_kv_transfer_cpu_us, &response.declared_input_bytes,
          &response.declared_output_bytes,
          nullptr,
          request.transfer_id,
          direct_device_import ? ipc_metadata.get() : nullptr);
    }
    if (!WriteAll(client, &response, sizeof(response))) {
      exit_status = 72;
      break;
    }
  }

  if (profiler.initialized) StopPostWarmupProfiler(&profiler);

#ifdef CRUISE_RESIDENT_DEVICE_TRANSFER_PLUGIN
  resident_epoch_destroy(engine);
  engine = nullptr;
  CloseDeviceTransferPlugin(&transfer_plugin);
#else
  resident_epoch_destroy(engine);
#endif
  close(client);
  close(listener);
  unlink(socket_path);
  return exit_status;
}
