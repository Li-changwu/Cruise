#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace {
using RtMemcpy = int32_t (*)(void *, uint64_t, const void *, uint64_t, int32_t);
using RtMemcpyAsync = int32_t (*)(void *, uint64_t, const void *, uint64_t,
                                  int32_t, void *);

int64_t RealtimeNs() {
  timespec value{};
  if (clock_gettime(CLOCK_REALTIME, &value) != 0) return -1;
  return static_cast<int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

int TraceFd() {
  static const int fd = []() {
    const char *path = std::getenv("ASCEND_RT_MEMCPY_TRACE_PATH");
    if (path == nullptr || std::strncmp(path, "/dev/shm/", 9) != 0) return -1;
    const int opened = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (opened < 0) return -1;
    struct stat state {};
    if (fstat(opened, &state) == 0 && state.st_size == 0) {
      constexpr char kHeader[] =
          "api\tpid\ttid\tstart_ns\tend_ns\tbytes\tdest_max\tkind\tstatus\n";
      const auto ignored = write(opened, kHeader, sizeof(kHeader) - 1);
      (void)ignored;
    }
    return opened;
  }();
  return fd;
}

RtMemcpy RealRtMemcpy() {
  static const auto function =
      reinterpret_cast<RtMemcpy>(dlsym(RTLD_NEXT, "rtMemcpy"));
  return function;
}

RtMemcpyAsync RealRtMemcpyAsync() {
  static const auto function =
      reinterpret_cast<RtMemcpyAsync>(dlsym(RTLD_NEXT, "rtMemcpyAsync"));
  return function;
}

void Record(const char *api, int64_t start_ns, int64_t end_ns, uint64_t bytes,
            uint64_t dest_max, int32_t kind, int32_t status) {
  const int fd = TraceFd();
  if (fd < 0) return;
  char line[256];
  const int length = std::snprintf(
      line, sizeof(line), "%s\t%d\t%ld\t%lld\t%lld\t%llu\t%llu\t%d\t%d\n", api,
      static_cast<int>(getpid()), static_cast<long>(syscall(SYS_gettid)),
      static_cast<long long>(start_ns), static_cast<long long>(end_ns),
      static_cast<unsigned long long>(bytes),
      static_cast<unsigned long long>(dest_max), kind, status);
  if (length > 0 && static_cast<size_t>(length) < sizeof(line)) {
    const auto ignored = write(fd, line, static_cast<size_t>(length));
    (void)ignored;
  }
}
}  // namespace

extern "C" int32_t rtMemcpy(void *dst, uint64_t dest_max, const void *src,
                            uint64_t count, int32_t kind) {
  const int64_t start_ns = RealtimeNs();
  const auto function = RealRtMemcpy();
  const int32_t status =
      function == nullptr ? -1 : function(dst, dest_max, src, count, kind);
  Record("rtMemcpy", start_ns, RealtimeNs(), count, dest_max, kind, status);
  return status;
}

extern "C" int32_t rtMemcpyAsync(void *dst, uint64_t dest_max, const void *src,
                                  uint64_t count, int32_t kind, void *stream) {
  const int64_t start_ns = RealtimeNs();
  const auto function = RealRtMemcpyAsync();
  const int32_t status = function == nullptr
                             ? -1
                             : function(dst, dest_max, src, count, kind, stream);
  Record("rtMemcpyAsync", start_ns, RealtimeNs(), count, dest_max, kind,
         status);
  return status;
}
