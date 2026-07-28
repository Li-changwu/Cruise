#include "register/op_def_registry.h"

namespace ops {

class Bf16Barrier : public OpDef {
public:
    explicit Bf16Barrier(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Bf16Barrier);

}  // namespace ops

