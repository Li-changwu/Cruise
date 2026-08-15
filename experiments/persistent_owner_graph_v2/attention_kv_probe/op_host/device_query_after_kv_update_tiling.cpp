#include <cstdint>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DeviceQueryAfterKvUpdateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, queryElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DeviceQueryAfterKvUpdate, DeviceQueryAfterKvUpdateTilingData)

struct DeviceQueryAfterKvUpdateCompileInfo {};

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  constexpr int64_t kQueryElements = 4LL * 28LL * 128LL;
  const auto *query = context->GetInputShape(0);
  const auto *ticket = context->GetInputShape(1);
  auto *raw = context->GetRawTilingData();
  if (query == nullptr || ticket == nullptr || raw == nullptr ||
      query->GetStorageShape().GetShapeSize() != kQueryElements ||
      ticket->GetStorageShape().GetShapeSize() != 9) {
    return ge::GRAPH_FAILED;
  }
  DeviceQueryAfterKvUpdateTilingData tiling;
  tiling.set_queryElements(static_cast<uint32_t>(kQueryElements));
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

IMPL_OP_OPTILING(DeviceQueryAfterKvUpdate)
    .Tiling(TilingFunc)
    .TilingParse<DeviceQueryAfterKvUpdateCompileInfo>(TilingParse);

}  // namespace optiling
