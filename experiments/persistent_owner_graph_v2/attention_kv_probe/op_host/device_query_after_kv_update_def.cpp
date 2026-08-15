#include "register/op_def_registry.h"

namespace ops {

class DeviceQueryAfterKvUpdate : public OpDef {
 public:
  explicit DeviceQueryAfterKvUpdate(const char *name) : OpDef(name) {
    this->Input("query").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
    this->Input("ticket").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    this->Output("ordered_query").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(DeviceQueryAfterKvUpdate);

}  // namespace ops
