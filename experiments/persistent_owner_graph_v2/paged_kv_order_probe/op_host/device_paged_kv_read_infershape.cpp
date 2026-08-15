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

static ge::graphStatus InferShape4DevicePagedKvRead(
    gert::InferShapeContext *context) {
  const gert::Shape *state = context->GetInputShape(0);
  const gert::Shape *metadata = context->GetInputShape(1);
  const gert::Shape *ticket = context->GetInputShape(2);
  gert::Shape *report = context->GetOutputShape(0);
  if (!IsTargetState(state) || metadata == nullptr || ticket == nullptr ||
      report == nullptr || metadata->GetDimNum() != 1 ||
      metadata->GetDim(0) != 5 || ticket->GetDimNum() != 1 ||
      ticket->GetDim(0) != 9) {
    return ge::GRAPH_FAILED;
  }
  report->SetDimNum(1);
  report->SetDim(0, 18);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DevicePagedKvRead(
    gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_INT32);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DevicePagedKvRead)
    .InferShape(InferShape4DevicePagedKvRead)
    .InferDataType(InferDataType4DevicePagedKvRead);

}  // namespace ops
