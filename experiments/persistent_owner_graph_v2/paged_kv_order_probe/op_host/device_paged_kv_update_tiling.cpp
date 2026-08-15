#include <cstdint>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DevicePagedKvUpdateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, stateElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DevicePagedKvUpdate, DevicePagedKvUpdateTilingData)

struct DevicePagedKvUpdateCompileInfo {};

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  constexpr int64_t kStateElements = 2LL * 12LL * 32LL * 128LL * 16LL;
  const auto *state = context->GetInputShape(0);
  auto *raw = context->GetRawTilingData();
  if (state == nullptr || raw == nullptr ||
      state->GetStorageShape().GetShapeSize() != kStateElements) {
    return ge::GRAPH_FAILED;
  }
  DevicePagedKvUpdateTilingData tiling;
  tiling.set_stateElements(static_cast<uint32_t>(kStateElements));
  tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
  raw->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(1);
  context->SetTilingKey(1);
  return context->SetScheduleMode(1);
}

static ge::graphStatus TilingParseFunc(gert::TilingParseContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DevicePagedKvUpdate)
    .Tiling(TilingFunc)
    .TilingParse<DevicePagedKvUpdateCompileInfo>(TilingParseFunc);

}  // namespace optiling
