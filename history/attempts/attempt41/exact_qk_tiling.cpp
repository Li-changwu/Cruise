#include <cstring>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_data.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ExactQkTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, m);
    TILING_DATA_FIELD_DEF(uint32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, m0);
    TILING_DATA_FIELD_DEF(uint32_t, k0);
    TILING_DATA_FIELD_DEF(uint32_t, n0);
    TILING_DATA_FIELD_DEF(uint32_t, mLoop);
    TILING_DATA_FIELD_DEF(uint32_t, kLoop);
    TILING_DATA_FIELD_DEF(uint32_t, nLoop);
    TILING_DATA_FIELD_DEF(uint32_t, coreLoop);
    TILING_DATA_FIELD_DEF(uint32_t, swizzlCount);
    TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, swizzlDirect);
    TILING_DATA_FIELD_DEF(uint32_t, splitk);
    TILING_DATA_FIELD_DEF(uint32_t, enShuffleK);
    TILING_DATA_FIELD_DEF(uint32_t, quantMode);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ExactQk0TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, m);
    TILING_DATA_FIELD_DEF(uint32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, m0);
    TILING_DATA_FIELD_DEF(uint32_t, k0);
    TILING_DATA_FIELD_DEF(uint32_t, n0);
    TILING_DATA_FIELD_DEF(uint32_t, mLoop);
    TILING_DATA_FIELD_DEF(uint32_t, kLoop);
    TILING_DATA_FIELD_DEF(uint32_t, nLoop);
    TILING_DATA_FIELD_DEF(uint32_t, coreLoop);
    TILING_DATA_FIELD_DEF(uint32_t, swizzlCount);
    TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, swizzlDirect);
    TILING_DATA_FIELD_DEF(uint32_t, splitk);
    TILING_DATA_FIELD_DEF(uint32_t, enShuffleK);
    TILING_DATA_FIELD_DEF(uint32_t, quantMode);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ExactQk1TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, m);
    TILING_DATA_FIELD_DEF(uint32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, m0);
    TILING_DATA_FIELD_DEF(uint32_t, k0);
    TILING_DATA_FIELD_DEF(uint32_t, n0);
    TILING_DATA_FIELD_DEF(uint32_t, mLoop);
    TILING_DATA_FIELD_DEF(uint32_t, kLoop);
    TILING_DATA_FIELD_DEF(uint32_t, nLoop);
    TILING_DATA_FIELD_DEF(uint32_t, coreLoop);
    TILING_DATA_FIELD_DEF(uint32_t, swizzlCount);
    TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, swizzlDirect);
    TILING_DATA_FIELD_DEF(uint32_t, splitk);
    TILING_DATA_FIELD_DEF(uint32_t, enShuffleK);
    TILING_DATA_FIELD_DEF(uint32_t, quantMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ExactQk, ExactQkTilingData)
REGISTER_TILING_DATA_CLASS(ExactQk_0, ExactQk0TilingData)
REGISTER_TILING_DATA_CLASS(ExactQk_1, ExactQk1TilingData)

struct ExactQkCompileInfo {};

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto raw = context->GetRawTilingData();
    if (raw == nullptr || raw->GetCapacity() < sizeof(pp_matmul::PpMatmulTilingData)) {
        return ge::GRAPH_FAILED;
    }

    pp_matmul::MatMulInfo info;
    info.batchSize = 28;
    info.m = 1;
    info.k = 128;
    info.n = 8;
    info.dtypeA = pp_matmul::TensorDType::TENSOR_DTYPE_BF16;
    info.dtypeB = pp_matmul::TensorDType::TENSOR_DTYPE_BF16;
    info.dtypeC = pp_matmul::TensorDType::TENSOR_DTYPE_BF16;
    info.formatA = pp_matmul::TensorFormat::TENSOR_FORMAT_ND;
    info.formatB = pp_matmul::TensorFormat::TENSOR_FORMAT_ND;
    info.formatC = pp_matmul::TensorFormat::TENSOR_FORMAT_ND;
    info.mmType = pp_matmul::MatMul::MatMulType::MATMUL_EIN_SUM;
    info.inDtype = 2.0F;
    info.outDtype = 2.0F;

    pp_matmul::HardwareInfo hardware;
    pp_matmul::PpMatmulTilingData tiling;
    uint32_t blockDim = 0;
    pp_matmul::GetPpMatmulTiling(info, hardware, blockDim, tiling);
    if (blockDim == 0) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &tiling, sizeof(tiling));
    raw->SetDataSize(sizeof(tiling));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(1);
    if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseFunc(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ExactQk)
    .Tiling(TilingFunc)
    .TilingParse<ExactQkCompileInfo>(TilingParseFunc);

}  // namespace optiling
