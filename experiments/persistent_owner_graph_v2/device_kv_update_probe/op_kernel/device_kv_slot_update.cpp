#define __aicore__ [aicore]
#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kMetadataElements = 4U;
constexpr uint32_t kReportElements = 9U;
constexpr uint16_t kLeftSentinel = 0x3e00U;
constexpr uint16_t kRightSentinel = 0xbe00U;
}  // namespace

extern "C" __global__ __aicore__ void device_kv_slot_update(
    GM_ADDR gm_cache, GM_ADDR gm_metadata, GM_ADDR gm_report,
    GM_ADDR workspace, GM_ADDR gm_tiling_data) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
  if (GetBlockIdx() != 0) return;

  GET_TILING_DATA(tiling_data, gm_tiling_data);
  const uint32_t cache_elements = tiling_data.cacheElements;

  GlobalTensor<uint16_t> cache;
  GlobalTensor<int32_t> metadata;
  GlobalTensor<int32_t> report;
  cache.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_cache),
                        cache_elements);
  metadata.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_metadata),
                           kMetadataElements);
  report.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_report),
                         kReportElements);

  const int32_t sequence = metadata.GetValue(0);
  const int32_t slot = metadata.GetValue(1);
  const uint16_t expected_before =
      static_cast<uint16_t>(metadata.GetValue(2) & 0xffff);
  const uint16_t requested_new =
      static_cast<uint16_t>(metadata.GetValue(3) & 0xffff);
  uint16_t observed_before = 0;
  uint16_t observed_after = 0;
  uint16_t left = 0;
  uint16_t right = 0;
  int32_t status = 0;

  if (slot <= 0 || static_cast<uint32_t>(slot + 1) >= cache_elements) {
    status |= 1;
  } else {
    observed_before = cache.GetValue(static_cast<uint32_t>(slot));
    left = cache.GetValue(static_cast<uint32_t>(slot - 1));
    right = cache.GetValue(static_cast<uint32_t>(slot + 1));
    if (observed_before != expected_before) status |= 2;
    if (left != kLeftSentinel || right != kRightSentinel) status |= 4;
    if (status == 0) {
      cache.SetValue(static_cast<uint32_t>(slot), requested_new);
      DataCacheCleanAndInvalid<uint16_t, CacheLine::ENTIRE_DATA_CACHE>(cache);
      observed_after = cache.GetValue(static_cast<uint32_t>(slot));
      if (observed_after != requested_new) status |= 8;
    }
  }

  report.SetValue(0, sequence);
  report.SetValue(1, slot);
  report.SetValue(2, static_cast<int32_t>(observed_before));
  report.SetValue(3, static_cast<int32_t>(observed_after));
  report.SetValue(4, static_cast<int32_t>(expected_before));
  report.SetValue(5, static_cast<int32_t>(requested_new));
  report.SetValue(6, static_cast<int32_t>(left));
  report.SetValue(7, static_cast<int32_t>(right));
  report.SetValue(8, status);
  DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(report);
}
