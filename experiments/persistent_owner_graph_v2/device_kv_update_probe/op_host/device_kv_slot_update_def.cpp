#include "register/op_def_registry.h"

namespace ops {

class DeviceKvSlotUpdate : public OpDef {
 public:
  explicit DeviceKvSlotUpdate(const char *name) : OpDef(name) {
    this->Input("cache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("metadata")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("report")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(DeviceKvSlotUpdate);

}  // namespace ops
