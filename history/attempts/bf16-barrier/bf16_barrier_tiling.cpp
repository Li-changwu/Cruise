#include "register/op_def_registry.h"

namespace optiling {

struct Bf16BarrierCompileInfo {};

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    context->SetBlockDim(1);
    context->SetTilingKey(1);
    return context->SetScheduleMode(1);
}

static ge::graphStatus TilingParseFunc(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Bf16Barrier)
    .Tiling(TilingFunc)
    .TilingParse<Bf16BarrierCompileInfo>(TilingParseFunc);

}  // namespace optiling

