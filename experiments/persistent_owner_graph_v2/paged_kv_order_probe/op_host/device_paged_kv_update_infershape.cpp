#include "register/op_def_registry.h"

namespace ops {
namespace {
bool IsTargetState(const gert::Shape *shape) {
  const int64_t dimensions[5] = {2, 12, 32, 128, 16};
  if (shape == nullptr || shape->GetDimNum() != 5) return false;
  for (size_t index = 0; index < 5; ++index) {
    if (shape->GetDim(index) != dimensions[index]) return false;
  }
  return true;
}
}  // namespace

static ge::graphStatus InferShape4DevicePagedKvUpdate(
    gert::InferShapeContext *context) {
  const gert::Shape *state = context->GetInputShape(0);
  const gert::Shape *metadata = context->GetInputShape(1);
  gert::Shape *ticket = context->GetOutputShape(0);
  if (!IsTargetState(state) || metadata == nullptr || ticket == nullptr ||
      metadata->GetDimNum() != 1 || metadata->GetDim(0) != 5) {
    return ge::GRAPH_FAILED;
  }
  ticket->SetDimNum(1);
  ticket->SetDim(0, 9);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DevicePagedKvUpdate(
    gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_INT32);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DevicePagedKvUpdate)
    .InferShape(InferShape4DevicePagedKvUpdate)
    .InferDataType(InferDataType4DevicePagedKvUpdate);

}  // namespace ops
