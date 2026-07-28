#include <cstdint>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

namespace {
constexpr int64_t kMaxElements = 32768;
}

BEGIN_TILING_DATA_DEF(Bf16MaterializeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, elementCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Bf16Materialize, Bf16MaterializeTilingData)

struct Bf16MaterializeCompileInfo {};

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *x = context->GetInputShape(0);
    auto *raw = context->GetRawTilingData();
    if (x == nullptr || raw == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t elements = x->GetStorageShape().GetShapeSize();
    if (elements <= 0 || elements > kMaxElements) {
        return ge::GRAPH_FAILED;
    }

    Bf16MaterializeTilingData tiling;
    tiling.set_elementCount(static_cast<uint32_t>(elements));
    if (tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(1);
    context->SetTilingKey(1);
    return context->SetScheduleMode(1);
}

static ge::graphStatus TilingParseFunc(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Bf16Materialize)
    .Tiling(TilingFunc)
    .TilingParse<Bf16MaterializeCompileInfo>(TilingParseFunc);

}  // namespace optiling
