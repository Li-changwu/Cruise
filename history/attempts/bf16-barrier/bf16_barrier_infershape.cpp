#include "register/op_def_registry.h"

namespace ops {

static ge::graphStatus InferShape4Bf16Barrier(gert::InferShapeContext *context)
{
    const gert::Shape *x = context->GetInputShape(0);
    gert::Shape *y = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr || x->GetDimNum() != 4 ||
        x->GetDim(0) != 1 || x->GetDim(1) != 28 ||
        x->GetDim(2) != 1 || x->GetDim(3) != 8) {
        return ge::GRAPH_FAILED;
    }
    *y = *x;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Bf16Barrier(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Bf16Barrier)
    .InferShape(InferShape4Bf16Barrier)
    .InferDataType(InferDataType4Bf16Barrier);

}  // namespace ops

