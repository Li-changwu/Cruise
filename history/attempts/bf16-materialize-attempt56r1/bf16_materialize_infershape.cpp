#include "register/op_def_registry.h"

namespace ops {

static ge::graphStatus InferShape4Bf16Materialize(gert::InferShapeContext *context)
{
    const gert::Shape *x = context->GetInputShape(0);
    gert::Shape *y = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr || x->GetShapeSize() <= 0 ||
        x->GetShapeSize() > 32768) {
        return ge::GRAPH_FAILED;
    }
    *y = *x;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Bf16Materialize(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Bf16Materialize)
    .InferShape(InferShape4Bf16Materialize)
    .InferDataType(InferDataType4Bf16Materialize);

}  // namespace ops
