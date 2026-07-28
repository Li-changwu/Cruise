#include "register/op_def_registry.h"

namespace ops {

static ge::graphStatus InferShape4ExactQk(gert::InferShapeContext *context)
{
    const gert::Shape *a = context->GetInputShape(0);
    const gert::Shape *b = context->GetInputShape(1);
    const gert::Shape *explicit_tiling = context->GetInputShape(2);
    gert::Shape *c = context->GetOutputShape(0);
    if (a == nullptr || b == nullptr || explicit_tiling == nullptr || c == nullptr ||
        a->GetDimNum() != 3 || b->GetDimNum() != 3 ||
        explicit_tiling->GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    if (a->GetDim(0) != 1 || a->GetDim(1) != 28 || a->GetDim(2) != 128 ||
        b->GetDim(0) != 28 || b->GetDim(1) != 128 || b->GetDim(2) != 8 ||
        explicit_tiling->GetDim(0) != 72) {
        return ge::GRAPH_FAILED;
    }
    c->SetDimNum(3);
    c->SetDim(0, 1);
    c->SetDim(1, 28);
    c->SetDim(2, 8);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ExactQk(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ExactQk)
    .InferShape(InferShape4ExactQk)
    .InferDataType(InferDataType4ExactQk);

}  // namespace ops
