#define __aicore__ [aicore]
#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBatchSize = 4U;
constexpr uint32_t kBlocksPerRow = 3U;
constexpr uint32_t kPhysicalBlocks = 12U;
constexpr uint32_t kPackedChannels = 32U;
constexpr uint32_t kBlockSize = 128U;
constexpr uint32_t kPackWidth = 16U;
constexpr uint32_t kFeatures = kPackedChannels * kPackWidth;
constexpr uint32_t kStateElements =
    2U * kPhysicalBlocks * kPackedChannels * kBlockSize * kPackWidth;
constexpr uint32_t kMetadataElements = 5U;
constexpr uint32_t kTicketElements = 9U;
constexpr uint32_t kUpdatedElements = 2U * kBatchSize * kFeatures;

__aicore__ inline uint32_t StateIndex(uint32_t cache_kind, uint32_t row,
                                     uint32_t position, uint32_t feature) {
  const uint32_t block = row * kBlocksPerRow + position / kBlockSize;
  const uint32_t offset = position % kBlockSize;
  const uint32_t channel = feature / kPackWidth;
  const uint32_t lane = feature % kPackWidth;
  return ((((cache_kind * kPhysicalBlocks + block) * kPackedChannels +
            channel) *
               kBlockSize +
           offset) *
              kPackWidth +
          lane);
}

__aicore__ inline uint16_t ExpectedBits(int32_t sequence,
                                        uint32_t cache_kind, uint32_t row,
                                        uint32_t feature) {
  return static_cast<uint16_t>(0x2000U +
                               static_cast<uint32_t>(sequence) * 0x1000U +
                               cache_kind * 0x0800U + row * 0x0200U + feature);
}
}  // namespace

extern "C" __global__ __aicore__ void device_paged_kv_update(
    GM_ADDR gm_state, GM_ADDR gm_metadata, GM_ADDR gm_ticket,
    GM_ADDR workspace, GM_ADDR gm_tiling_data) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
  if (GetBlockIdx() != 0) return;

  GET_TILING_DATA(tiling_data, gm_tiling_data);
  if (tiling_data.stateElements != kStateElements) return;

  GlobalTensor<uint16_t> state;
  GlobalTensor<int32_t> metadata;
  GlobalTensor<int32_t> ticket;
  state.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_state),
                        kStateElements);
  metadata.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_metadata),
                           kMetadataElements);
  ticket.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_ticket),
                         kTicketElements);

  const int32_t sequence = metadata.GetValue(0);
  const uint32_t required_positions[kBatchSize] = {0U, 127U, 128U, 383U};
  uint32_t positions[kBatchSize] = {0U, 0U, 0U, 0U};
  int32_t status = 0;
  if (sequence != 1 && sequence != 2) status |= 1;
  for (uint32_t row = 0; row < kBatchSize; ++row) {
    const int32_t raw_position = metadata.GetValue(row + 1U);
    if (raw_position < 0 ||
        static_cast<uint32_t>(raw_position) >= kBlocksPerRow * kBlockSize) {
      status |= 2;
    } else {
      positions[row] = static_cast<uint32_t>(raw_position);
      if (positions[row] != required_positions[row]) status |= 4;
    }
  }

  if (status == 0) {
    for (uint32_t cache_kind = 0; cache_kind < 2U; ++cache_kind) {
      for (uint32_t row = 0; row < kBatchSize; ++row) {
        for (uint32_t feature = 0; feature < kFeatures; ++feature) {
          const uint32_t index =
              StateIndex(cache_kind, row, positions[row], feature);
          const uint16_t expected_before =
              sequence == 1
                  ? 0U
                  : ExpectedBits(sequence - 1, cache_kind, row, feature);
          if (state.GetValue(index) != expected_before) status |= 8;
        }
      }
    }
  }

  if (status == 0) {
    for (uint32_t cache_kind = 0; cache_kind < 2U; ++cache_kind) {
      for (uint32_t row = 0; row < kBatchSize; ++row) {
        for (uint32_t feature = 0; feature < kFeatures; ++feature) {
          state.SetValue(StateIndex(cache_kind, row, positions[row], feature),
                         ExpectedBits(sequence, cache_kind, row, feature));
        }
      }
    }
    DataCacheCleanAndInvalid<uint16_t, CacheLine::ENTIRE_DATA_CACHE>(state);
  }

  ticket.SetValue(0, sequence);
  ticket.SetValue(1, status);
  ticket.SetValue(2, static_cast<int32_t>(kUpdatedElements));
  for (uint32_t row = 0; row < kBatchSize; ++row) {
    ticket.SetValue(row + 3U,
                    static_cast<int32_t>(row * kBlocksPerRow * kBlockSize +
                                         positions[row]));
  }
  ticket.SetValue(7, status == 0 ? static_cast<int32_t>(ExpectedBits(
                                           sequence, 0U, 0U, 0U))
                                 : 0);
  ticket.SetValue(8, status == 0 ? static_cast<int32_t>(ExpectedBits(
                                           sequence, 1U, 3U, kFeatures - 1U))
                                 : 0);
  DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(ticket);
}
