#include <cstdint>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DeviceKvSlotUpdateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, cacheElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DeviceKvSlotUpdate, DeviceKvSlotUpdateTilingData)

struct DeviceKvSlotUpdateCompileInfo {};

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  const auto *cache = context->GetInputShape(0);
  auto *raw = context->GetRawTilingData();
  if (cache == nullptr || raw == nullptr ||
      cache->GetStorageShape().GetShapeSize() != 4096) {
    return ge::GRAPH_FAILED;
  }
  DeviceKvSlotUpdateTilingData tiling;
  tiling.set_cacheElements(4096U);
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

IMPL_OP_OPTILING(DeviceKvSlotUpdate)
    .Tiling(TilingFunc)
    .TilingParse<DeviceKvSlotUpdateCompileInfo>(TilingParseFunc);

}  // namespace optiling
