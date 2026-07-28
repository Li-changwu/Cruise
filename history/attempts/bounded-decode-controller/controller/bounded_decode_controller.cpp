#include <cstdint>
#include <memory>
#include <vector>

#include "flow_func/flow_func_log.h"
#include "flow_func/meta_flow_func.h"

namespace FlowFunc {
namespace {
constexpr size_t kModelTensorCount = 4;
constexpr size_t kInputCount = 5;
constexpr size_t kOutputCount = 5;
constexpr size_t kInputControlElements = 6;
constexpr size_t kOutputControlElements = 12;
constexpr int32_t kRunModelTimeoutMs = 100000;
constexpr int32_t kMaxSteps = 32;
constexpr int32_t kSyntheticVocabSize = 151936;

enum ControlInputIndex : size_t {
  kMaxStepsIndex = 0,
  kEosTokenIndex = 1,
  kEosAfterStepIndex = 2,
  kGraphSwitchStepIndex = 3,
  kTokenSeedIndex = 4,
  kTokenStrideIndex = 5,
};

enum ControlOutputIndex : size_t {
  kExecutedStepsIndex = 6,
  kFinalTokenIndex = 7,
  kFinishReasonIndex = 8,
  kGraph0CallsIndex = 9,
  kGraph1CallsIndex = 10,
  kFinalPositionIndex = 11,
};

enum FinishReason : int32_t {
  kFinishEos = 1,
  kFinishMaxSteps = 2,
};

bool IsInt32Tensor(const std::shared_ptr<FlowMsg> &msg, size_t elements) {
  if (msg == nullptr || msg->GetRetCode() != FLOW_FUNC_SUCCESS) {
    return false;
  }
  auto *tensor = msg->GetTensor();
  return tensor != nullptr &&
         tensor->GetDataType() == TensorDataType::DT_INT32 &&
         tensor->GetDataSize() == elements * sizeof(int32_t) &&
         tensor->GetData() != nullptr;
}

int32_t ReadScalarPosition(const std::shared_ptr<FlowMsg> &msg,
                           int32_t &position) {
  if (msg == nullptr || msg->GetRetCode() != FLOW_FUNC_SUCCESS) {
    return FLOW_FUNC_FAILED;
  }
  auto *tensor = msg->GetTensor();
  if (tensor == nullptr || tensor->GetElementCnt() != 1 ||
      tensor->GetData() == nullptr) {
    return FLOW_FUNC_FAILED;
  }
  if (tensor->GetDataType() == TensorDataType::DT_INT64) {
    const auto value = *static_cast<const int64_t *>(tensor->GetData());
    if (value < 0 || value > INT32_MAX) {
      return FLOW_FUNC_FAILED;
    }
    position = static_cast<int32_t>(value);
    return FLOW_FUNC_SUCCESS;
  }
  if (tensor->GetDataType() == TensorDataType::DT_INT32) {
    position = *static_cast<const int32_t *>(tensor->GetData());
    return FLOW_FUNC_SUCCESS;
  }
  return FLOW_FUNC_FAILED;
}

bool HasCapacityForBoundedRun(
    const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
    int32_t initial_position, int32_t max_steps) {
  if (input_msgs[0] == nullptr || input_msgs[2] == nullptr ||
      input_msgs[3] == nullptr) {
    return false;
  }
  auto *hidden = input_msgs[0]->GetTensor();
  auto *key_cache = input_msgs[2]->GetTensor();
  auto *value_cache = input_msgs[3]->GetTensor();
  if (hidden == nullptr || key_cache == nullptr || value_cache == nullptr) {
    return false;
  }
  const auto &hidden_shape = hidden->GetShape();
  const auto &key_shape = key_cache->GetShape();
  const auto &value_shape = value_cache->GetShape();
  if (hidden_shape.empty() || key_shape.size() < 3 || value_shape.size() < 3 ||
      key_shape[2] != value_shape[2] || initial_position < 0) {
    return false;
  }
  const int64_t required_position =
      static_cast<int64_t>(initial_position) + max_steps;
  return required_position <= hidden_shape[0] &&
         required_position <= key_shape[2];
}
}  // namespace

class BoundedDecodeController : public MetaFlowFunc {
 public:
  int32_t Init() override { return FLOW_FUNC_SUCCESS; }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) override {
    if (input_msgs.size() != kInputCount ||
        !IsInt32Tensor(input_msgs[4], kInputControlElements)) {
      FLOW_FUNC_LOG_ERROR("Invalid bounded-decode inputs, count[%zu].",
                          input_msgs.size());
      return FLOW_FUNC_FAILED;
    }

    const auto *input_control = static_cast<const int32_t *>(
        input_msgs[4]->GetTensor()->GetData());
    const int32_t max_steps = input_control[kMaxStepsIndex];
    const int32_t eos_token = input_control[kEosTokenIndex];
    const int32_t eos_after_step = input_control[kEosAfterStepIndex];
    const int32_t graph_switch_step = input_control[kGraphSwitchStepIndex];
    const int32_t token_seed = input_control[kTokenSeedIndex];
    const int32_t token_stride = input_control[kTokenStrideIndex];
    if (max_steps < 1 || max_steps > kMaxSteps || eos_token < 0 ||
        eos_token >= kSyntheticVocabSize || eos_after_step < 0 ||
        eos_after_step > max_steps || graph_switch_step < 0 ||
        graph_switch_step > max_steps || token_seed < 0 ||
        token_seed >= kSyntheticVocabSize || token_stride < 0 ||
        token_stride >= kSyntheticVocabSize) {
      FLOW_FUNC_LOG_ERROR(
          "Invalid control max[%d] eos[%d] eos_after[%d] switch[%d] "
          "seed[%d] stride[%d].",
          max_steps, eos_token, eos_after_step, graph_switch_step, token_seed,
          token_stride);
      return FLOW_FUNC_FAILED;
    }
    int32_t initial_position = -1;
    if (ReadScalarPosition(input_msgs[1], initial_position) !=
            FLOW_FUNC_SUCCESS ||
        !HasCapacityForBoundedRun(input_msgs, initial_position, max_steps)) {
      FLOW_FUNC_LOG_ERROR(
          "Bounded run exceeds staged hidden/KV capacity, position[%d] "
          "max_steps[%d].",
          initial_position, max_steps);
      return FLOW_FUNC_FAILED;
    }

    auto control_msg = context_->AllocTensorMsg(
        {static_cast<int64_t>(kOutputControlElements)},
        TensorDataType::DT_INT32);
    if (!IsInt32Tensor(control_msg, kOutputControlElements)) {
      FLOW_FUNC_LOG_ERROR("Failed to allocate bounded-decode control output.");
      return FLOW_FUNC_FAILED;
    }
    auto *control =
        static_cast<int32_t *>(control_msg->GetTensor()->GetData());
    for (size_t index = 0; index < kInputControlElements; ++index) {
      control[index] = input_control[index];
    }
    for (size_t index = kInputControlElements;
         index < kOutputControlElements; ++index) {
      control[index] = 0;
    }

    std::vector<std::shared_ptr<FlowMsg>> current_inputs = {
        input_msgs[0], input_msgs[1], input_msgs[2], input_msgs[3]};
    std::vector<std::shared_ptr<FlowMsg>> outputs;
    int32_t executed_steps = 0;
    int32_t final_token = token_seed;
    int32_t finish_reason = kFinishMaxSteps;
    int32_t graph0_calls = 0;
    int32_t graph1_calls = 0;

    while (executed_steps < max_steps) {
      const bool use_graph0 = executed_steps < graph_switch_step;
      const char *model_key = use_graph0 ? "decode_graph_0" : "decode_graph_1";
      outputs.clear();
      const auto ret = context_->RunFlowModel(
          model_key, current_inputs, outputs, kRunModelTimeoutMs);
      if (ret != FLOW_FUNC_SUCCESS || outputs.size() != kModelTensorCount) {
        FLOW_FUNC_LOG_ERROR(
            "RunFlowModel key[%s] failed at step[%d], ret[%d], outputs[%zu].",
            model_key, executed_steps, ret, outputs.size());
        return ret == FLOW_FUNC_SUCCESS ? FLOW_FUNC_FAILED : ret;
      }

      ++executed_steps;
      if (use_graph0) {
        ++graph0_calls;
      } else {
        ++graph1_calls;
      }
      const int64_t generated =
          static_cast<int64_t>(token_seed) +
          static_cast<int64_t>(token_stride) * executed_steps;
      final_token = static_cast<int32_t>(generated % kSyntheticVocabSize);
      if (eos_after_step > 0 && executed_steps == eos_after_step) {
        final_token = eos_token;
      }
      if (final_token == eos_token) {
        finish_reason = kFinishEos;
        break;
      }
      if (executed_steps >= max_steps) {
        finish_reason = kFinishMaxSteps;
        break;
      }
      current_inputs = {input_msgs[0], outputs[3], outputs[1], outputs[2]};
    }

    int32_t final_position = -1;
    if (outputs.size() != kModelTensorCount ||
        ReadScalarPosition(outputs[3], final_position) != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("Invalid final position after[%d] steps.",
                          executed_steps);
      return FLOW_FUNC_FAILED;
    }
    control[kExecutedStepsIndex] = executed_steps;
    control[kFinalTokenIndex] = final_token;
    control[kFinishReasonIndex] = finish_reason;
    control[kGraph0CallsIndex] = graph0_calls;
    control[kGraph1CallsIndex] = graph1_calls;
    control[kFinalPositionIndex] = final_position;

    for (size_t index = 0; index < kModelTensorCount; ++index) {
      const auto ret = context_->SetOutput(index, outputs[index]);
      if (ret != FLOW_FUNC_SUCCESS) {
        return ret;
      }
    }
    const auto ret = context_->SetOutput(4, control_msg);
    if (ret != FLOW_FUNC_SUCCESS) {
      return ret;
    }
    FLOW_FUNC_LOG_INFO(
        "Bounded decode complete: steps[%d] token[%d] reason[%d] "
        "graph0[%d] graph1[%d] position[%d].",
        executed_steps, final_token, finish_reason, graph0_calls, graph1_calls,
        final_position);
    return FLOW_FUNC_SUCCESS;
  }
};

REGISTER_FLOW_FUNC("bounded_decode_controller", BoundedDecodeController);
}  // namespace FlowFunc
