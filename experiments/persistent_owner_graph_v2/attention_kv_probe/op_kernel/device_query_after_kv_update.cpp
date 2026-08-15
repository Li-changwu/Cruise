#define __aicore__ [aicore]
#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void device_query_after_kv_update(
    GM_ADDR gm_query, GM_ADDR gm_ticket, GM_ADDR gm_output, GM_ADDR workspace,
    GM_ADDR gm_tiling_data) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
  if (GetBlockIdx() != 0) return;

  GET_TILING_DATA(tiling_data, gm_tiling_data);
  GlobalTensor<uint16_t> query;
  GlobalTensor<int32_t> ticket;
  GlobalTensor<uint16_t> output;
  query.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_query),
                        tiling_data.queryElements);
  ticket.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(gm_ticket), 9U);
  output.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_output),
                         tiling_data.queryElements);
  const bool valid =
      (ticket.GetValue(0) == 1 || ticket.GetValue(0) == 2) &&
      ticket.GetValue(1) == 0 && ticket.GetValue(2) == 4096;
  for (uint32_t index = 0; index < tiling_data.queryElements; ++index) {
    output.SetValue(index, valid ? query.GetValue(index) : 0U);
  }
  DataCacheCleanAndInvalid<uint16_t, CacheLine::ENTIRE_DATA_CACHE>(output);
}
