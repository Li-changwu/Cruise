#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "ge/ge_data_flow_api.h"

namespace {
using Memcpy = int32_t (*)(void *, uint64_t, const void *, uint64_t, int32_t);
using MemcpyAsync = int32_t (*)(void *, uint64_t, const void *, uint64_t,
                                int32_t, void *);
using MemcpyAsyncEx = int32_t (*)(void *, uint64_t, const void *, uint64_t,
                                  int32_t, void *, void *);
using MemcpyAsyncWithCfg = int32_t (*)(void *, uint64_t, const void *,
                                       uint64_t, int32_t, void *, uint32_t);
using MemcpyAsyncWithOffset = int32_t (*)(
    void **, uint64_t, uint64_t, const void **, uint64_t, uint64_t, int32_t,
    void *);
using Memcpy2d = int32_t (*)(void *, uint64_t, const void *, uint64_t,
                             uint64_t, uint64_t, int32_t);
using Memcpy2dAsync = int32_t (*)(void *, uint64_t, const void *, uint64_t,
                                  uint64_t, uint64_t, int32_t, void *);
using RtsMemcpy = int32_t (*)(void *, uint64_t, const void *, uint64_t,
                              int32_t, void *);
using RtsMemcpyAsync = int32_t (*)(void *, uint64_t, const void *, uint64_t,
                                   int32_t, void *, void *);

struct MemLocation {
  uint32_t id;
  int32_t type;
};

struct MemcpyBatchAttr {
  MemLocation dst;
  MemLocation src;
  uint8_t reserved[16];
};

using RtsMemcpyBatch = int32_t (*)(void **, void **, size_t *, size_t,
                                   MemcpyBatchAttr *, size_t *, size_t,
                                   size_t *);
using RtsMemcpyBatchAsync = int32_t (*)(
    void **, size_t *, void **, size_t *, size_t, MemcpyBatchAttr *, size_t *,
    size_t, size_t *, void *);
using RtsSetMemcpyDesc = int32_t (*)(void *, int32_t, void *, void *, size_t,
                                     void *);
using RtsMemcpyAsyncWithDesc = int32_t (*)(void *, int32_t, void *, void *);
using MbufAlloc = int32_t (*)(void **, uint64_t);
using MbufBuild = int32_t (*)(void *, uint64_t, void **);
using MbufSizeOperation = int32_t (*)(void *, uint64_t);
using MbufGetSize = int32_t (*)(void *, uint64_t *);
using BuffGet = int32_t (*)(const void *, void *, uint64_t);
using DflowFeed = uint32_t (*)(void *, uint32_t, const std::vector<uint32_t> *,
                               const std::vector<ge::Tensor> *,
                               const ge::DataFlowInfo *, int32_t);
using DflowFetch = uint32_t (*)(void *, uint32_t,
                                const std::vector<uint32_t> *,
                                std::vector<ge::Tensor> *, ge::DataFlowInfo *,
                                int32_t);

constexpr char kDflowFeedSymbol[] =
    "_ZN2ge16DFlowSessionImpl17FeedDataFlowGraphEjRKSt6vectorIjSaIjEERKS1_"
    "INS_6TensorESaIS6_EERKNS_12DataFlowInfoEi";
constexpr char kDflowFetchSymbol[] =
    "_ZN2ge16DFlowSessionImpl18FetchDataFlowGraphEjRKSt6vectorIjSaIjEERS1_"
    "INS_6TensorESaIS6_EERNS_12DataFlowInfoEi";

struct DescInfo {
  uint64_t bytes;
  int32_t kind;
};

thread_local uint32_t trace_depth = 0;
std::mutex desc_mutex;
std::unordered_map<void *, DescInfo> desc_info;

class TraceScope {
 public:
  TraceScope() : outer_(trace_depth++ == 0) {}
  ~TraceScope() { --trace_depth; }
  bool outer() const { return outer_; }

 private:
  bool outer_;
};

template <typename Function>
Function Resolve(const char *name) {
  return reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
}

int64_t RealtimeNs() {
  timespec value{};
  if (clock_gettime(CLOCK_REALTIME, &value) != 0) return -1;
  return static_cast<int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

int TraceFd() {
  static const int fd = []() {
    const char *path = std::getenv("ASCEND_RT_MEMCPY_TRACE_PATH");
    if (path == nullptr || std::strncmp(path, "/dev/shm/", 9) != 0) return -1;
    const int opened =
        open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
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

int32_t BatchKind(const MemcpyBatchAttr *attrs, const size_t *attrs_idxs,
                  size_t num_attrs, size_t copy_index) {
  if (attrs == nullptr || attrs_idxs == nullptr || num_attrs == 0) return 8;
  size_t attr_index = 0;
  for (size_t index = 1; index < num_attrs; ++index) {
    if (attrs_idxs[index] > copy_index) break;
    attr_index = index;
  }
  const int32_t src = attrs[attr_index].src.type;
  const int32_t dst = attrs[attr_index].dst.type;
  if (src == 0 && dst == 0) return 0;
  if (src == 0 && dst == 1) return 1;
  if (src == 1 && dst == 0) return 2;
  if (src == 1 && dst == 1) return 3;
  return 8;
}

void RecordBatch(const char *api, int64_t start_ns, int64_t end_ns,
                 const size_t *dest_maxs, const size_t *sizes, size_t count,
                 const MemcpyBatchAttr *attrs, const size_t *attrs_idxs,
                 size_t num_attrs, int32_t status) {
  if (sizes == nullptr) return;
  for (size_t index = 0; index < count; ++index) {
    const uint64_t bytes = sizes[index];
    const uint64_t dest_max = dest_maxs == nullptr ? bytes : dest_maxs[index];
    Record(api, start_ns, end_ns, bytes, dest_max,
           BatchKind(attrs, attrs_idxs, num_attrs, index), status);
  }
}
}  // namespace

extern "C" int32_t rtMemcpy(void *dst, uint64_t dest_max, const void *src,
                            uint64_t count, int32_t kind) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<Memcpy>("rtMemcpy");
  const int32_t status =
      function == nullptr ? -1 : function(dst, dest_max, src, count, kind);
  if (scope.outer())
    Record("rtMemcpy", start_ns, RealtimeNs(), count, dest_max, kind, status);
  return status;
}

extern "C" int32_t rtMemcpyEx(void *dst, uint64_t dest_max, const void *src,
                              uint64_t count, int32_t kind) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<Memcpy>("rtMemcpyEx");
  const int32_t status =
      function == nullptr ? -1 : function(dst, dest_max, src, count, kind);
  if (scope.outer())
    Record("rtMemcpyEx", start_ns, RealtimeNs(), count, dest_max, kind, status);
  return status;
}

extern "C" int32_t rtMemcpyAsync(void *dst, uint64_t dest_max,
                                 const void *src, uint64_t count, int32_t kind,
                                 void *stream) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<MemcpyAsync>("rtMemcpyAsync");
  const int32_t status = function == nullptr
                             ? -1
                             : function(dst, dest_max, src, count, kind, stream);
  if (scope.outer())
    Record("rtMemcpyAsync", start_ns, RealtimeNs(), count, dest_max, kind,
           status);
  return status;
}

extern "C" int32_t rtMemcpyAsyncWithoutCheckKind(
    void *dst, uint64_t dest_max, const void *src, uint64_t count, int32_t kind,
    void *stream) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function =
      Resolve<MemcpyAsync>("rtMemcpyAsyncWithoutCheckKind");
  const int32_t status = function == nullptr
                             ? -1
                             : function(dst, dest_max, src, count, kind, stream);
  if (scope.outer())
    Record("rtMemcpyAsyncWithoutCheckKind", start_ns, RealtimeNs(), count,
           dest_max, kind, status);
  return status;
}

extern "C" int32_t rtMemcpyAsyncEx(void *dst, uint64_t dest_max,
                                   const void *src, uint64_t count,
                                   int32_t kind, void *stream, void *config) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<MemcpyAsyncEx>("rtMemcpyAsyncEx");
  const int32_t status =
      function == nullptr
          ? -1
          : function(dst, dest_max, src, count, kind, stream, config);
  if (scope.outer())
    Record("rtMemcpyAsyncEx", start_ns, RealtimeNs(), count, dest_max, kind,
           status);
  return status;
}

extern "C" int32_t rtMemcpyAsyncWithCfg(
    void *dst, uint64_t dest_max, const void *src, uint64_t count, int32_t kind,
    void *stream, uint32_t qos_cfg) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function =
      Resolve<MemcpyAsyncWithCfg>("rtMemcpyAsyncWithCfg");
  const int32_t status = function == nullptr
                             ? -1
                             : function(dst, dest_max, src, count, kind, stream,
                                        qos_cfg);
  if (scope.outer())
    Record("rtMemcpyAsyncWithCfg", start_ns, RealtimeNs(), count, dest_max,
           kind, status);
  return status;
}

extern "C" int32_t rtMemcpyAsyncWithCfgV2(
    void *dst, uint64_t dest_max, const void *src, uint64_t count, int32_t kind,
    void *stream, const void *config) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<MemcpyAsyncEx>("rtMemcpyAsyncWithCfgV2");
  const int32_t status =
      function == nullptr
          ? -1
          : function(dst, dest_max, src, count, kind, stream,
                     const_cast<void *>(config));
  if (scope.outer())
    Record("rtMemcpyAsyncWithCfgV2", start_ns, RealtimeNs(), count, dest_max,
           kind, status);
  return status;
}

extern "C" int32_t rtMemcpyHostTask(void *dst, uint64_t dest_max,
                                    const void *src, uint64_t count,
                                    int32_t kind, void *stream) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<MemcpyAsync>("rtMemcpyHostTask");
  const int32_t status = function == nullptr
                             ? -1
                             : function(dst, dest_max, src, count, kind, stream);
  if (scope.outer())
    Record("rtMemcpyHostTask", start_ns, RealtimeNs(), count, dest_max, kind,
           status);
  return status;
}

extern "C" int32_t rtMemcpyAsyncWithOffset(
    void **dst, uint64_t dest_max, uint64_t dst_offset, const void **src,
    uint64_t count, uint64_t src_offset, int32_t kind, void *stream) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function =
      Resolve<MemcpyAsyncWithOffset>("rtMemcpyAsyncWithOffset");
  const int32_t status =
      function == nullptr
          ? -1
          : function(dst, dest_max, dst_offset, src, count, src_offset, kind,
                     stream);
  if (scope.outer())
    Record("rtMemcpyAsyncWithOffset", start_ns, RealtimeNs(), count, dest_max,
           kind, status);
  return status;
}

extern "C" int32_t rtMemcpy2d(void *dst, uint64_t dst_pitch, const void *src,
                              uint64_t src_pitch, uint64_t width,
                              uint64_t height, int32_t kind) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<Memcpy2d>("rtMemcpy2d");
  const int32_t status = function == nullptr
                             ? -1
                             : function(dst, dst_pitch, src, src_pitch, width,
                                        height, kind);
  if (scope.outer())
    Record("rtMemcpy2d", start_ns, RealtimeNs(), width * height,
           dst_pitch * height, kind, status);
  return status;
}

extern "C" int32_t rtMemcpy2dAsync(
    void *dst, uint64_t dst_pitch, const void *src, uint64_t src_pitch,
    uint64_t width, uint64_t height, int32_t kind, void *stream) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<Memcpy2dAsync>("rtMemcpy2dAsync");
  const int32_t status =
      function == nullptr
          ? -1
          : function(dst, dst_pitch, src, src_pitch, width, height, kind,
                     stream);
  if (scope.outer())
    Record("rtMemcpy2dAsync", start_ns, RealtimeNs(), width * height,
           dst_pitch * height, kind, status);
  return status;
}

extern "C" int32_t rtsMemcpy(void *dst, uint64_t dest_max, const void *src,
                             uint64_t count, int32_t kind, void *config) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<RtsMemcpy>("rtsMemcpy");
  const int32_t status = function == nullptr
                             ? -1
                             : function(dst, dest_max, src, count, kind, config);
  if (scope.outer())
    Record("rtsMemcpy", start_ns, RealtimeNs(), count, dest_max, kind, status);
  return status;
}

extern "C" int32_t rtsMemcpyAsync(void *dst, uint64_t dest_max,
                                  const void *src, uint64_t count,
                                  int32_t kind, void *config, void *stream) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<RtsMemcpyAsync>("rtsMemcpyAsync");
  const int32_t status =
      function == nullptr
          ? -1
          : function(dst, dest_max, src, count, kind, config, stream);
  if (scope.outer())
    Record("rtsMemcpyAsync", start_ns, RealtimeNs(), count, dest_max, kind,
           status);
  return status;
}

extern "C" int32_t rtsMemcpyBatch(
    void **dsts, void **srcs, size_t *sizes, size_t count,
    MemcpyBatchAttr *attrs, size_t *attrs_idxs, size_t num_attrs,
    size_t *fail_idx) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<RtsMemcpyBatch>("rtsMemcpyBatch");
  const int32_t status =
      function == nullptr
          ? -1
          : function(dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs,
                     fail_idx);
  const int64_t end_ns = scope.outer() ? RealtimeNs() : 0;
  if (scope.outer())
    RecordBatch("rtsMemcpyBatch", start_ns, end_ns, nullptr, sizes, count,
                attrs, attrs_idxs, num_attrs, status);
  return status;
}

extern "C" int32_t rtsMemcpyBatchAsync(
    void **dsts, size_t *dest_maxs, void **srcs, size_t *sizes, size_t count,
    MemcpyBatchAttr *attrs, size_t *attrs_idxs, size_t num_attrs,
    size_t *fail_idx, void *stream) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function =
      Resolve<RtsMemcpyBatchAsync>("rtsMemcpyBatchAsync");
  const int32_t status =
      function == nullptr
          ? -1
          : function(dsts, dest_maxs, srcs, sizes, count, attrs, attrs_idxs,
                     num_attrs, fail_idx, stream);
  const int64_t end_ns = scope.outer() ? RealtimeNs() : 0;
  if (scope.outer())
    RecordBatch("rtsMemcpyBatchAsync", start_ns, end_ns, dest_maxs, sizes,
                count, attrs, attrs_idxs, num_attrs, status);
  return status;
}

extern "C" int32_t rtsSetMemcpyDesc(void *desc, int32_t kind, void *src,
                                    void *dst, size_t count, void *config) {
  TraceScope scope;
  static const auto function = Resolve<RtsSetMemcpyDesc>("rtsSetMemcpyDesc");
  const int32_t status = function == nullptr
                             ? -1
                             : function(desc, kind, src, dst, count, config);
  if (scope.outer() && status == 0 && desc != nullptr) {
    std::lock_guard<std::mutex> lock(desc_mutex);
    desc_info[desc] = DescInfo{count, kind};
  }
  return status;
}

extern "C" int32_t rtsMemcpyAsyncWithDesc(void *desc, int32_t kind,
                                          void *config, void *stream) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function =
      Resolve<RtsMemcpyAsyncWithDesc>("rtsMemcpyAsyncWithDesc");
  const int32_t status =
      function == nullptr ? -1 : function(desc, kind, config, stream);
  if (scope.outer()) {
    DescInfo info{0, kind};
    {
      std::lock_guard<std::mutex> lock(desc_mutex);
      const auto found = desc_info.find(desc);
      if (found != desc_info.end()) info = found->second;
    }
    Record("rtsMemcpyAsyncWithDesc", start_ns, RealtimeNs(), info.bytes,
           info.bytes, kind, status);
  }
  return status;
}

extern "C" int32_t rtMbufAlloc(void **mbuf, uint64_t size) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<MbufAlloc>("rtMbufAlloc");
  const int32_t status = function == nullptr ? -1 : function(mbuf, size);
  if (scope.outer())
    Record("rtMbufAlloc", start_ns, RealtimeNs(), size, size, -1, status);
  return status;
}

extern "C" int32_t rtMbufBuild(void *buffer, uint64_t size, void **mbuf) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<MbufBuild>("rtMbufBuild");
  const int32_t status =
      function == nullptr ? -1 : function(buffer, size, mbuf);
  if (scope.outer())
    Record("rtMbufBuild", start_ns, RealtimeNs(), size, size, -1, status);
  return status;
}

extern "C" int32_t rtMbufSetDataLen(void *mbuf, uint64_t size) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function =
      Resolve<MbufSizeOperation>("rtMbufSetDataLen");
  const int32_t status = function == nullptr ? -1 : function(mbuf, size);
  if (scope.outer())
    Record("rtMbufSetDataLen", start_ns, RealtimeNs(), size, size, -1,
           status);
  return status;
}

extern "C" int32_t rtMbufGetBuffSize(void *mbuf, uint64_t *size) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<MbufGetSize>("rtMbufGetBuffSize");
  const int32_t status = function == nullptr ? -1 : function(mbuf, size);
  const uint64_t bytes = status == 0 && size != nullptr ? *size : 0;
  if (scope.outer())
    Record("rtMbufGetBuffSize", start_ns, RealtimeNs(), bytes, bytes, -1,
           status);
  return status;
}

extern "C" int32_t rtBuffGet(const void *mbuf, void *buffer, uint64_t size) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<BuffGet>("rtBuffGet");
  const int32_t status =
      function == nullptr ? -1 : function(mbuf, buffer, size);
  if (scope.outer())
    Record("rtBuffGet", start_ns, RealtimeNs(), size, size, -1, status);
  return status;
}

extern "C" int32_t rtBuffConfirm(void *buffer, uint64_t size) {
  TraceScope scope;
  const int64_t start_ns = scope.outer() ? RealtimeNs() : 0;
  static const auto function = Resolve<MbufSizeOperation>("rtBuffConfirm");
  const int32_t status = function == nullptr ? -1 : function(buffer, size);
  if (scope.outer())
    Record("rtBuffConfirm", start_ns, RealtimeNs(), size, size, -1, status);
  return status;
}

extern "C" uint32_t TraceDflowFeed(
    void *self, uint32_t graph_id, const std::vector<uint32_t> *indexes,
    const std::vector<ge::Tensor> *inputs, const ge::DataFlowInfo *flow_info,
    int32_t timeout) asm(
    "_ZN2ge16DFlowSessionImpl17FeedDataFlowGraphEjRKSt6vectorIjSaIjEERKS1_"
    "INS_6TensorESaIS6_EERKNS_12DataFlowInfoEi");

extern "C" uint32_t TraceDflowFeed(
    void *self, uint32_t graph_id, const std::vector<uint32_t> *indexes,
    const std::vector<ge::Tensor> *inputs, const ge::DataFlowInfo *flow_info,
    int32_t timeout) {
  const int64_t start_ns = RealtimeNs();
  static const auto function = Resolve<DflowFeed>(kDflowFeedSymbol);
  const uint32_t status =
      function == nullptr
          ? static_cast<uint32_t>(-1)
          : function(self, graph_id, indexes, inputs, flow_info, timeout);
  const int64_t end_ns = RealtimeNs();
  if (inputs != nullptr) {
    for (const auto &tensor : *inputs) {
      const uint64_t bytes = tensor.GetSize();
      Record("FeedDataFlowGraphTensor", start_ns, end_ns, bytes, bytes, -1,
             static_cast<int32_t>(status));
    }
  }
  return status;
}

extern "C" uint32_t TraceDflowFetch(
    void *self, uint32_t graph_id, const std::vector<uint32_t> *indexes,
    std::vector<ge::Tensor> *outputs, ge::DataFlowInfo *flow_info,
    int32_t timeout) asm(
    "_ZN2ge16DFlowSessionImpl18FetchDataFlowGraphEjRKSt6vectorIjSaIjEERS1_"
    "INS_6TensorESaIS6_EERNS_12DataFlowInfoEi");

extern "C" uint32_t TraceDflowFetch(
    void *self, uint32_t graph_id, const std::vector<uint32_t> *indexes,
    std::vector<ge::Tensor> *outputs, ge::DataFlowInfo *flow_info,
    int32_t timeout) {
  const int64_t start_ns = RealtimeNs();
  static const auto function = Resolve<DflowFetch>(kDflowFetchSymbol);
  const uint32_t status =
      function == nullptr
          ? static_cast<uint32_t>(-1)
          : function(self, graph_id, indexes, outputs, flow_info, timeout);
  const int64_t end_ns = RealtimeNs();
  if (outputs != nullptr) {
    for (const auto &tensor : *outputs) {
      const uint64_t bytes = tensor.GetSize();
      Record("FetchDataFlowGraphTensor", start_ns, end_ns, bytes, bytes, -1,
             static_cast<int32_t>(status));
    }
  }
  return status;
}
