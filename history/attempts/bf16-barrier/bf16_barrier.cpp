#define __aicore__ [aicore]
#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kElements = 1U * 28U * 1U * 8U;
constexpr uint32_t kBufferBytes = kElements * sizeof(uint16_t);
}  // namespace

extern "C" __global__ __aicore__ void bf16_barrier(
    GM_ADDR gm_x, GM_ADDR gm_y, GM_ADDR workspace, GM_ADDR gm_tiling_data)
{
    (void)workspace;
    (void)gm_tiling_data;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) {
        return;
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kBufferBytes);
    auto local = buffer.Get<uint16_t>();

    GlobalTensor<uint16_t> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_x), kElements);
    DataCopy(local, input, kElements);
    event_t mte2ToS = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(mte2ToS);
    WaitFlag<HardEvent::MTE2_S>(mte2ToS);
    for (uint32_t index = 0; index < kElements; ++index) {
        const uint16_t value = local.GetValue(index);
        local.SetValue(index, value);
    }

    GlobalTensor<uint16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gm_y), kElements);
    event_t sToMte3 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(sToMte3);
    WaitFlag<HardEvent::S_MTE3>(sToMte3);
    DataCopy(output, local, kElements);
}

