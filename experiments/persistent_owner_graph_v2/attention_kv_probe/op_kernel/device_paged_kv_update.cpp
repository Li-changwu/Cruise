#define __aicore__ [aicore]
#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBatch = 4U;
constexpr uint32_t kBlocksPerRow = 3U;
constexpr uint32_t kPhysicalBlocks = 12U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kPacksPerHead = 8U;
constexpr uint32_t kBlockSize = 128U;
constexpr uint32_t kPackWidth = 16U;
constexpr uint32_t kHeadDim = kPacksPerHead * kPackWidth;
constexpr uint32_t kFeatures = kKvHeads * kHeadDim;
constexpr uint32_t kCacheElements =
    kPhysicalBlocks * kKvHeads * kPacksPerHead * kBlockSize * kPackWidth;
constexpr uint32_t kUpdatedElements = 2U * kBatch * kFeatures;

__aicore__ inline uint32_t CacheIndex(uint32_t row, uint32_t position,
                                     uint32_t feature) {
  const uint32_t block = row * kBlocksPerRow + position / kBlockSize;
  const uint32_t offset = position % kBlockSize;
  const uint32_t head = feature / kHeadDim;
  const uint32_t dim = feature % kHeadDim;
  const uint32_t pack = dim / kPackWidth;
  const uint32_t lane = dim % kPackWidth;
  return (((((block * kKvHeads + head) * kPacksPerHead + pack) * kBlockSize +
            offset) *
               kPackWidth) +
          lane);
}

__aicore__ inline uint16_t SequenceBits(int32_t sequence) {
  return sequence == 1 ? 0x3f80U : 0x4000U;
}
}  // namespace

extern "C" __global__ __aicore__ void device_paged_kv_update(
    GM_ADDR gm_key, GM_ADDR gm_value, GM_ADDR gm_metadata, GM_ADDR gm_ticket,
    GM_ADDR workspace, GM_ADDR gm_tiling_data) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
  if (GetBlockIdx() != 0) return;

  GET_TILING_DATA(tiling_data, gm_tiling_data);
  if (tiling_data.cacheElements != kCacheElements) return;
  GlobalTensor<uint16_t> key;
  GlobalTensor<uint16_t> value;
  GlobalTensor<int32_t> metadata;
  GlobalTensor<int32_t> ticket;
  key.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_key), kCacheElements);
  value.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_value), kCacheElements);
  metadata.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_metadata), 5U);
  ticket.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_ticket), 9U);

  const int32_t sequence = metadata.GetValue(0);
  const uint32_t required[kBatch] = {0U, 127U, 128U, 383U};
  uint32_t positions[kBatch] = {0U, 0U, 0U, 0U};
  int32_t status = (sequence == 1 || sequence == 2) ? 0 : 1;
  for (uint32_t row = 0; row < kBatch; ++row) {
    const int32_t position = metadata.GetValue(row + 1U);
    if (position < 0 || static_cast<uint32_t>(position) >= 384U) {
      status |= 2;
    } else {
      positions[row] = static_cast<uint32_t>(position);
      if (positions[row] != required[row]) status |= 4;
    }
  }

  const uint16_t before = sequence == 1 ? 0U : SequenceBits(sequence - 1);
  if (status == 0) {
    for (uint32_t row = 0; row < kBatch; ++row) {
      for (uint32_t feature = 0; feature < kFeatures; ++feature) {
        const uint32_t index = CacheIndex(row, positions[row], feature);
        if (key.GetValue(index) != before || value.GetValue(index) != before) {
          status |= 8;
        }
      }
    }
  }
  if (status == 0) {
    const uint16_t bits = SequenceBits(sequence);
    for (uint32_t row = 0; row < kBatch; ++row) {
      for (uint32_t feature = 0; feature < kFeatures; ++feature) {
        const uint32_t index = CacheIndex(row, positions[row], feature);
        key.SetValue(index, bits);
        value.SetValue(index, bits);
      }
    }
    DataCacheCleanAndInvalid<uint16_t, CacheLine::ENTIRE_DATA_CACHE>(key);
    DataCacheCleanAndInvalid<uint16_t, CacheLine::ENTIRE_DATA_CACHE>(value);
  }

  ticket.SetValue(0, sequence);
  ticket.SetValue(1, status);
  ticket.SetValue(2, static_cast<int32_t>(kUpdatedElements));
  for (uint32_t row = 0; row < kBatch; ++row) {
    ticket.SetValue(row + 3U, static_cast<int32_t>(positions[row]));
  }
  ticket.SetValue(7, status == 0 ? SequenceBits(sequence) : 0);
  ticket.SetValue(8, status == 0 ? SequenceBits(sequence) : 0);
  DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(ticket);
}
