#include "register/op_impl_registry.h"

namespace {
ge::graphStatus InferShape(gert::InferShapeContext *context) {
  auto *output = context->GetOutputShape(0);
  if (output == nullptr) return ge::GRAPH_FAILED;
  *output = gert::Shape({9});
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(DevicePagedKvUpdate).InferShape(InferShape);
