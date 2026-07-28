#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "resident_epoch_bridge.h"

namespace {
constexpr uint32_t kRequestMagic = 0x71317131U;
constexpr uint32_t kResponseMagic = 0x71327132U;
constexpr uint16_t kProtocolVersion = 3;
constexpr uint16_t kExecute = 1;
constexpr uint16_t kWarmUp = 2;
constexpr uint16_t kShutdown = 3;
constexpr int32_t kBatchSize = 4;
constexpr int32_t kMaxEpochSteps = 8;

#pragma pack(push, 1)
struct Request {
  uint32_t magic;
  uint16_t version;
  uint16_t operation;
  int32_t request_count;
  int32_t max_steps;
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
  int64_t wall_us;
  int64_t native_cpu_us;
  int64_t declared_input_bytes;
  int64_t declared_output_bytes;
  int32_t executed[kBatchSize];
  int32_t row_generations[kBatchSize];
  int64_t token_ids[kBatchSize * kMaxEpochSteps];
};
#pragma pack(pop)

static_assert(sizeof(Request) == 128, "resident epoch request ABI changed");
static_assert(sizeof(Response) == 344, "resident epoch response ABI changed");

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
}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr,
                 "usage: %s SOCKET AIR GRAPH_CONFIG FUNC_CONFIG WEIGHTS TILING\n",
                 argv[0]);
    return 64;
  }
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
  Response startup = EmptyResponse(engine == nullptr ? 100 + create_status : 0);
  if (!WriteAll(client, &startup, sizeof(startup)) || engine == nullptr) {
    if (engine != nullptr) resident_epoch_destroy(engine);
    close(client);
    close(listener);
    unlink(socket_path);
    return engine == nullptr ? 100 + create_status : 67;
  }

  int exit_status = 0;
  while (true) {
    Request request{};
    if (!ReadAll(client, &request, sizeof(request))) {
      exit_status = 68;
      break;
    }
    Response response = EmptyResponse(0);
    if (request.magic != kRequestMagic ||
        request.version != kProtocolVersion) {
      response.transport_status = 69;
    } else if (request.operation == kShutdown) {
      if (!WriteAll(client, &response, sizeof(response))) exit_status = 70;
      break;
    } else if (request.operation != kExecute &&
               request.operation != kWarmUp) {
      response.transport_status = 71;
    } else {
      response.transport_status = resident_epoch_execute(
          engine, request.request_count, request.max_steps, request.token_ids,
          request.positions, request.sequence_lengths, request.eos_token_ids,
          request.row_generations, response.token_ids, response.executed,
          response.row_generations, &response.model_calls,
          &response.device_status, &response.feed_calls,
          &response.fetch_calls, &response.wall_us, &response.native_cpu_us,
          &response.declared_input_bytes, &response.declared_output_bytes);
    }
    if (!WriteAll(client, &response, sizeof(response))) {
      exit_status = 72;
      break;
    }
  }

  resident_epoch_destroy(engine);
  close(client);
  close(listener);
  unlink(socket_path);
  return exit_status;
}
