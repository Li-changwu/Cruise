#include "register/op_def_registry.h"

namespace ops {

class Bf16Materialize : public OpDef {
public:
    explicit Bf16Materialize(const char *name) : OpDef(name)
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

OP_ADD(Bf16Materialize);

}  // namespace ops
