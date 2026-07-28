#define __aicore__ [aicore]
#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kMaxElements = 4U * 28U * 1U * 8U;
constexpr uint32_t kElementsPerBlock = 16U;
constexpr uint32_t kBufferBytes = kMaxElements * sizeof(uint16_t);
}  // namespace

extern "C" __global__ __aicore__ void bf16_barrier(
    GM_ADDR gm_x, GM_ADDR gm_y, GM_ADDR workspace, GM_ADDR gm_tiling_data)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) {
        return;
    }

    GET_TILING_DATA(tiling_data, gm_tiling_data);
    const uint32_t elements = tiling_data.elementCount;
    if (elements == 0 || elements > kMaxElements) {
        return;
    }

    GlobalTensor<uint16_t> input;
    GlobalTensor<uint16_t> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_x), elements);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_y), elements);

    const uint32_t aligned_elements =
        elements / kElementsPerBlock * kElementsPerBlock;
    if (aligned_elements != 0) {
        TPipe pipe;
        TBuf<TPosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, kBufferBytes);
        auto local = buffer.Get<uint16_t>();
        DataCopy(local, input, aligned_elements);
        event_t mte2_to_s = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2_to_s);
        WaitFlag<HardEvent::MTE2_S>(mte2_to_s);
        for (uint32_t index = 0; index < aligned_elements; ++index) {
            const uint16_t value = local.GetValue(index);
            local.SetValue(index, value);
        }
        event_t s_to_mte3 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(s_to_mte3);
        WaitFlag<HardEvent::S_MTE3>(s_to_mte3);
        DataCopy(output, local, aligned_elements);
    }
    for (uint32_t index = aligned_elements; index < elements; ++index) {
        output.SetValue(index, input.GetValue(index));
    }
    if (aligned_elements != elements) {
        DataCacheCleanAndInvalid<uint16_t, CacheLine::ENTIRE_DATA_CACHE>(output);
    }
}
