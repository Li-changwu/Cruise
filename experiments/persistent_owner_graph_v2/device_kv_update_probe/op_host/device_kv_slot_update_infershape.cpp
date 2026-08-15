#include "register/op_def_registry.h"

namespace ops {

static ge::graphStatus InferShape4DeviceKvSlotUpdate(
    gert::InferShapeContext *context) {
  const gert::Shape *cache = context->GetInputShape(0);
  const gert::Shape *metadata = context->GetInputShape(1);
  gert::Shape *report = context->GetOutputShape(0);
  if (cache == nullptr || metadata == nullptr || report == nullptr ||
      cache->GetDimNum() != 1 || cache->GetDim(0) != 4096 ||
      metadata->GetDimNum() != 1 || metadata->GetDim(0) != 4) {
    return ge::GRAPH_FAILED;
  }
  report->SetDimNum(1);
  report->SetDim(0, 9);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DeviceKvSlotUpdate(
    gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_INT32);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DeviceKvSlotUpdate)
    .InferShape(InferShape4DeviceKvSlotUpdate)
    .InferDataType(InferDataType4DeviceKvSlotUpdate);

}  // namespace ops
