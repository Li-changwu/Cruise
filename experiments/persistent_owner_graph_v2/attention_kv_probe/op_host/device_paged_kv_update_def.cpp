#include "register/op_def_registry.h"

namespace ops {

class DevicePagedKvUpdate : public OpDef {
 public:
  explicit DevicePagedKvUpdate(const char *name) : OpDef(name) {
    this->Input("key_cache").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
    this->Input("value_cache").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
    this->Input("metadata").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    this->Output("ticket").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(DevicePagedKvUpdate);

}  // namespace ops
