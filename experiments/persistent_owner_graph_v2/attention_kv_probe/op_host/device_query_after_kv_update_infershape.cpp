#include "register/op_impl_registry.h"

namespace {
ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const auto *input = context->GetInputShape(0);
  auto *output = context->GetOutputShape(0);
  if (input == nullptr || output == nullptr) return ge::GRAPH_FAILED;
  *output = *input;
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(DeviceQueryAfterKvUpdate).InferShape(InferShape);
