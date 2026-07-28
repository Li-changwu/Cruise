#include <cstdint>
#include <memory>
#include <vector>

#include "flow_func/flow_func_log.h"
#include "flow_func/meta_flow_func.h"

namespace FlowFunc {
class Bf16Echo : public MetaFlowFunc {
 public:
  int32_t Init() override { return FLOW_FUNC_SUCCESS; }

  int32_t Proc(
      const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) override {
    if (input_msgs.size() != 1 || input_msgs[0] == nullptr ||
        input_msgs[0]->GetRetCode() != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("BF16 echo expected one successful input.");
      return FLOW_FUNC_FAILED;
    }
    auto *tensor = input_msgs[0]->GetTensor();
    if (tensor == nullptr) {
      FLOW_FUNC_LOG_ERROR("BF16 echo input tensor is null.");
      return FLOW_FUNC_FAILED;
    }
    FLOW_FUNC_LOG_INFO("BF16 echo received dtype[%d] elements[%ld] bytes[%zu].",
                       static_cast<int>(tensor->GetDataType()),
                       static_cast<long>(tensor->GetElementCnt()),
                       tensor->GetDataSize());
    if (tensor->GetDataType() != TensorDataType::DT_BF16 ||
        tensor->GetElementCnt() != 16 || tensor->GetDataSize() != 32 ||
        tensor->GetData() == nullptr) {
      FLOW_FUNC_LOG_ERROR("BF16 echo input contract mismatch.");
      return FLOW_FUNC_FAILED;
    }
    return context_->SetOutput(0, input_msgs[0]);
  }
};

REGISTER_FLOW_FUNC("bf16_echo", Bf16Echo);
}  // namespace FlowFunc
