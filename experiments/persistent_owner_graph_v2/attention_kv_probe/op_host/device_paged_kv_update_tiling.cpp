#include <cstdint>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DevicePagedKvUpdateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, cacheElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DevicePagedKvUpdate, DevicePagedKvUpdateTilingData)

struct DevicePagedKvUpdateCompileInfo {};

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  constexpr int64_t kCacheElements = 12LL * 4LL * 8LL * 128LL * 16LL;
  const auto *key = context->GetInputShape(0);
  const auto *value = context->GetInputShape(1);
  const auto *metadata = context->GetInputShape(2);
  auto *raw = context->GetRawTilingData();
  if (key == nullptr || value == nullptr || metadata == nullptr || raw == nullptr ||
      key->GetStorageShape().GetShapeSize() != kCacheElements ||
      value->GetStorageShape().GetShapeSize() != kCacheElements ||
      metadata->GetStorageShape().GetShapeSize() != 5) {
    return ge::GRAPH_FAILED;
  }
  DevicePagedKvUpdateTilingData tiling;
  tiling.set_cacheElements(static_cast<uint32_t>(kCacheElements));
  tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
  raw->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(1);
  context->SetTilingKey(1);
  return context->SetScheduleMode(1);
}

static ge::graphStatus TilingParse(gert::TilingParseContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DevicePagedKvUpdate)
    .Tiling(TilingFunc)
    .TilingParse<DevicePagedKvUpdateCompileInfo>(TilingParse);

}  // namespace optiling
