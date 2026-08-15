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
constexpr uint32_t kReportElements = 18U;
constexpr uint32_t kUpdatedElements = 2U * kBatchSize * kFeatures;
constexpr uint32_t kChecksumModulus = 65521U;

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

extern "C" __global__ __aicore__ void device_paged_kv_read(
    GM_ADDR gm_state, GM_ADDR gm_metadata, GM_ADDR gm_ticket,
    GM_ADDR gm_report, GM_ADDR workspace, GM_ADDR gm_tiling_data) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
  if (GetBlockIdx() != 0) return;

  GET_TILING_DATA(tiling_data, gm_tiling_data);
  if (tiling_data.stateElements != kStateElements) return;

  GlobalTensor<uint16_t> state;
  GlobalTensor<int32_t> metadata;
  GlobalTensor<int32_t> ticket;
  GlobalTensor<int32_t> report;
  state.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_state),
                        kStateElements);
  metadata.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_metadata),
                           kMetadataElements);
  ticket.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_ticket),
                         kTicketElements);
  report.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_report),
                         kReportElements);
  DataCacheCleanAndInvalid<uint16_t, CacheLine::ENTIRE_DATA_CACHE>(state);

  const int32_t sequence = metadata.GetValue(0);
  uint32_t positions[kBatchSize] = {0U, 0U, 0U, 0U};
  int32_t dependency_status = 0;
  if (ticket.GetValue(0) != sequence || ticket.GetValue(1) != 0 ||
      ticket.GetValue(2) != static_cast<int32_t>(kUpdatedElements)) {
    dependency_status = 1;
  }
  for (uint32_t row = 0; row < kBatchSize; ++row) {
    positions[row] = static_cast<uint32_t>(metadata.GetValue(row + 1U));
    const int32_t logical_slot =
        static_cast<int32_t>(row * kBlocksPerRow * kBlockSize + positions[row]);
    if (ticket.GetValue(row + 3U) != logical_slot) dependency_status = 1;
  }

  uint32_t mismatch_count = 0U;
  uint32_t actual_checksum = 0U;
  uint32_t expected_checksum = 0U;
  for (uint32_t cache_kind = 0; cache_kind < 2U; ++cache_kind) {
    for (uint32_t row = 0; row < kBatchSize; ++row) {
      for (uint32_t feature = 0; feature < kFeatures; ++feature) {
        const uint16_t actual = state.GetValue(
            StateIndex(cache_kind, row, positions[row], feature));
        const uint16_t expected =
            ExpectedBits(sequence, cache_kind, row, feature);
        if (actual != expected) ++mismatch_count;
        actual_checksum = (actual_checksum + actual) % kChecksumModulus;
        expected_checksum = (expected_checksum + expected) % kChecksumModulus;
      }
    }
  }
  int32_t status = dependency_status != 0 ? 1 : 0;
  if (mismatch_count != 0U) status |= 2;
  if (actual_checksum != expected_checksum) status |= 4;

  report.SetValue(0, sequence);
  report.SetValue(1, ticket.GetValue(0));
  report.SetValue(2, ticket.GetValue(1));
  report.SetValue(3, ticket.GetValue(2));
  report.SetValue(4, static_cast<int32_t>(kUpdatedElements));
  report.SetValue(5, static_cast<int32_t>(mismatch_count));
  report.SetValue(6, static_cast<int32_t>(actual_checksum));
  report.SetValue(7, static_cast<int32_t>(expected_checksum));
  for (uint32_t row = 0; row < kBatchSize; ++row) {
    report.SetValue(row + 8U, ticket.GetValue(row + 3U));
  }
  report.SetValue(12, static_cast<int32_t>(state.GetValue(
                          StateIndex(0U, 0U, positions[0], 0U))));
  report.SetValue(13, static_cast<int32_t>(state.GetValue(StateIndex(
                          0U, 3U, positions[3], kFeatures - 1U))));
  report.SetValue(14, static_cast<int32_t>(state.GetValue(
                          StateIndex(1U, 0U, positions[0], 0U))));
  report.SetValue(15, static_cast<int32_t>(state.GetValue(StateIndex(
                          1U, 3U, positions[3], kFeatures - 1U))));
  report.SetValue(16, dependency_status);
  report.SetValue(17, status);
  DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(report);
}
