#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "flow_func/flow_func_log.h"
#include "flow_func/meta_flow_func.h"

namespace FlowFunc {
namespace {
constexpr size_t kInputCount = 9;
constexpr size_t kOutputCount = 6;
constexpr size_t kDecoderOutputCount = 4;
constexpr size_t kControlInputElements = 4;
constexpr size_t kControlOutputElements = 12;
constexpr int32_t kMaxEpochSteps = 8;
constexpr int32_t kLogicalCapacity = 8;
constexpr int32_t kPhysicalBlocks = 2;
constexpr int32_t kBlockSize = 128;
constexpr int32_t kVocabSize = 152064;
constexpr int32_t kRunModelTimeoutMs = 300000;
constexpr int64_t kCacheElements = 28LL * 2 * 128 * 4 * 128;

enum InputIndex : size_t {
  kTokenInput = 0,
  kPositionInput = 1,
  kSequenceLengthInput = 2,
  kKeyCacheInput = 3,
  kSlotMappingInput = 4,
  kBlockTableInput = 5,
  kValueCacheInput = 6,
  kTilingInput = 7,
  kControlInput = 8,
};

enum ControlInputIndex : size_t {
  kMaxStepsIndex = 0,
  kEosTokenIndex = 1,
  kSamplingModeIndex = 2,
  kGraphVariantIndex = 3,
};

enum ControlOutputIndex : size_t {
  kExecutedStepsIndex = 4,
  kFinishReasonIndex = 5,
  kStatusIndex = 6,
  kFinalTokenIndex = 7,
  kFinalPositionIndex = 8,
  kFinalSequenceLengthIndex = 9,
  kModelCallsIndex = 10,
  kFallbackRequiredIndex = 11,
};

enum FinishReason : int32_t {
  kFinishNone = 0,
  kFinishEos = 1,
  kFinishMaxSteps = 2,
  kFinishFallback = 3,
};

enum Status : int32_t {
  kStatusOk = 0,
  kStatusInvalidMetadata = 101,
  kStatusCapacityExceeded = 102,
  kStatusInvalidBlockTable = 103,
  kStatusSlotMismatch = 104,
  kStatusUnsupportedSampling = 105,
  kStatusUnsupportedGraph = 106,
  kStatusModelError = 107,
  kStatusInvalidModelOutput = 108,
  kStatusNonFiniteLogits = 109,
  kStatusPositionProgress = 110,
  kStatusAllocationFailure = 111,
};

bool IsTensor(const std::shared_ptr<FlowMsg> &msg, TensorDataType dtype,
              int64_t elements) {
  if (msg == nullptr || msg->GetRetCode() != FLOW_FUNC_SUCCESS) {
    return false;
  }
  auto *tensor = msg->GetTensor();
  return tensor != nullptr && tensor->GetDataType() == dtype &&
         tensor->GetElementCnt() == elements && tensor->GetData() != nullptr;
}

bool ReadInt64(const std::shared_ptr<FlowMsg> &msg, int64_t &value) {
  if (!IsTensor(msg, TensorDataType::DT_INT64, 1)) {
    return false;
  }
  value = *static_cast<const int64_t *>(msg->GetTensor()->GetData());
  return true;
}

bool ReadInt32(const std::shared_ptr<FlowMsg> &msg, int32_t &value) {
  if (!IsTensor(msg, TensorDataType::DT_INT32, 1)) {
    return false;
  }
  value = *static_cast<const int32_t *>(msg->GetTensor()->GetData());
  return true;
}

int32_t ComputeSlot(const int32_t *block_table, int64_t position) {
  if (position < 0 || position >= kLogicalCapacity) {
    return -1;
  }
  const int32_t logical_block = static_cast<int32_t>(position / kBlockSize);
  const int32_t offset = static_cast<int32_t>(position % kBlockSize);
  const int32_t physical_block = block_table[logical_block];
  if (physical_block < 0 || physical_block >= kPhysicalBlocks) {
    return -1;
  }
  return physical_block * kBlockSize + offset;
}

bool ArgmaxFinite(const std::shared_ptr<FlowMsg> &msg, int64_t &token) {
  if (!IsTensor(msg, TensorDataType::DT_FLOAT, kVocabSize)) {
    return false;
  }
  const auto *logits = static_cast<const float *>(msg->GetTensor()->GetData());
  if (!std::isfinite(logits[0])) {
    return false;
  }
  float best = logits[0];
  int64_t best_index = 0;
  for (int64_t index = 1; index < kVocabSize; ++index) {
    if (!std::isfinite(logits[index])) {
      return false;
    }
    if (logits[index] > best) {
      best = logits[index];
      best_index = index;
    }
  }
  token = best_index;
  return true;
}
}  // namespace

class G4bResidentEpoch : public MetaFlowFunc {
 public:
  int32_t Init() override { return FLOW_FUNC_SUCCESS; }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) override {
    if (input_msgs.size() != kInputCount) {
      FLOW_FUNC_LOG_ERROR("Invalid G4b input count[%zu].", input_msgs.size());
      return FLOW_FUNC_FAILED;
    }

    auto logits_history = context_->AllocTensorMsg(
        {kMaxEpochSteps, 1, 1, kVocabSize}, TensorDataType::DT_FLOAT);
    auto token_history = context_->AllocTensorMsg(
        {kMaxEpochSteps}, TensorDataType::DT_INT64);
    auto control_output = context_->AllocTensorMsg(
        {static_cast<int64_t>(kControlOutputElements)}, TensorDataType::DT_INT32);
    if (!IsTensor(logits_history, TensorDataType::DT_FLOAT,
                  static_cast<int64_t>(kMaxEpochSteps) * kVocabSize) ||
        !IsTensor(token_history, TensorDataType::DT_INT64, kMaxEpochSteps) ||
        !IsTensor(control_output, TensorDataType::DT_INT32,
                  kControlOutputElements)) {
      FLOW_FUNC_LOG_ERROR("Failed to allocate G4b epoch outputs.");
      return FLOW_FUNC_FAILED;
    }
    std::memset(logits_history->GetTensor()->GetData(), 0,
                logits_history->GetTensor()->GetDataSize());
    auto *tokens =
        static_cast<int64_t *>(token_history->GetTensor()->GetData());
    for (int32_t index = 0; index < kMaxEpochSteps; ++index) {
      tokens[index] = -1;
    }
    auto *control =
        static_cast<int32_t *>(control_output->GetTensor()->GetData());
    for (size_t index = 0; index < kControlOutputElements; ++index) {
      control[index] = 0;
    }

    int64_t initial_position = -1;
    int32_t initial_sequence_length = -1;
    int32_t initial_slot = -1;
    int64_t initial_token = -1;
    if (!IsTensor(input_msgs[kTokenInput], TensorDataType::DT_INT64, 1) ||
        !ReadInt64(input_msgs[kTokenInput], initial_token) ||
        !ReadInt64(input_msgs[kPositionInput], initial_position) ||
        !ReadInt32(input_msgs[kSequenceLengthInput], initial_sequence_length) ||
        !ReadInt32(input_msgs[kSlotMappingInput], initial_slot) ||
        !IsTensor(input_msgs[kBlockTableInput], TensorDataType::DT_INT32, 2) ||
        !IsTensor(input_msgs[kKeyCacheInput], TensorDataType::DT_BF16,
                  kCacheElements) ||
        !IsTensor(input_msgs[kValueCacheInput], TensorDataType::DT_BF16,
                  kCacheElements) ||
        !IsTensor(input_msgs[kTilingInput], TensorDataType::DT_UINT8, 72) ||
        !IsTensor(input_msgs[kControlInput], TensorDataType::DT_INT32,
                  kControlInputElements)) {
      FLOW_FUNC_LOG_ERROR("Invalid G4b tensor metadata.");
      return FLOW_FUNC_FAILED;
    }

    const auto *input_control = static_cast<const int32_t *>(
        input_msgs[kControlInput]->GetTensor()->GetData());
    for (size_t index = 0; index < kControlInputElements; ++index) {
      control[index] = input_control[index];
    }
    const int32_t max_steps = input_control[kMaxStepsIndex];
    const int32_t eos_token = input_control[kEosTokenIndex];
    const int32_t sampling_mode = input_control[kSamplingModeIndex];
    const int32_t graph_variant = input_control[kGraphVariantIndex];
    const auto *block_table = static_cast<const int32_t *>(
        input_msgs[kBlockTableInput]->GetTensor()->GetData());

    auto emit = [&](int32_t status, int32_t executed, int32_t reason,
                    int64_t final_token, int64_t final_position,
                    int32_t final_sequence_length, int32_t model_calls,
                    const std::shared_ptr<FlowMsg> &key,
                    const std::shared_ptr<FlowMsg> &value,
                    const std::shared_ptr<FlowMsg> &position) -> int32_t {
      control[kExecutedStepsIndex] = executed;
      control[kFinishReasonIndex] = reason;
      control[kStatusIndex] = status;
      control[kFinalTokenIndex] = static_cast<int32_t>(final_token);
      control[kFinalPositionIndex] = static_cast<int32_t>(final_position);
      control[kFinalSequenceLengthIndex] = final_sequence_length;
      control[kModelCallsIndex] = model_calls;
      control[kFallbackRequiredIndex] = status == kStatusOk ? 0 : 1;
      const std::vector<std::shared_ptr<FlowMsg>> outputs = {
          logits_history, token_history, key, value, position, control_output};
      for (size_t index = 0; index < outputs.size(); ++index) {
        const auto ret = context_->SetOutput(index, outputs[index]);
        if (ret != FLOW_FUNC_SUCCESS) {
          return ret;
        }
      }
      return FLOW_FUNC_SUCCESS;
    };

    auto fallback = [&](int32_t status, int32_t executed, int32_t model_calls,
                        int64_t final_token) -> int32_t {
      FLOW_FUNC_LOG_ERROR(
          "G4b fallback status[%d] executed[%d] calls[%d] position[%ld].",
          status, executed, model_calls, static_cast<long>(initial_position));
      return emit(status, executed, kFinishFallback, final_token,
                  initial_position, initial_sequence_length, model_calls,
                  input_msgs[kKeyCacheInput], input_msgs[kValueCacheInput],
                  input_msgs[kPositionInput]);
    };

    if (initial_token < 0 || initial_token >= kVocabSize ||
        initial_position < 0 || initial_sequence_length != initial_position + 1 ||
        eos_token < 0 || eos_token >= kVocabSize || max_steps < 1) {
      return fallback(kStatusInvalidMetadata, 0, 0, -1);
    }
    if (sampling_mode != 0) {
      return fallback(kStatusUnsupportedSampling, 0, 0, -1);
    }
    if (graph_variant != 0) {
      return fallback(kStatusUnsupportedGraph, 0, 0, -1);
    }
    if (max_steps > kMaxEpochSteps ||
        initial_position + max_steps > kLogicalCapacity) {
      return fallback(kStatusCapacityExceeded, 0, 0, -1);
    }
    if (block_table[0] < 0 || block_table[0] >= kPhysicalBlocks ||
        block_table[1] < 0 || block_table[1] >= kPhysicalBlocks ||
        block_table[0] == block_table[1]) {
      return fallback(kStatusInvalidBlockTable, 0, 0, -1);
    }
    if (ComputeSlot(block_table, initial_position) != initial_slot) {
      return fallback(kStatusSlotMismatch, 0, 0, -1);
    }

    auto current_token = input_msgs[kTokenInput];
    auto current_position = input_msgs[kPositionInput];
    auto current_sequence_length = input_msgs[kSequenceLengthInput];
    auto current_slot = input_msgs[kSlotMappingInput];
    auto current_key = input_msgs[kKeyCacheInput];
    auto current_value = input_msgs[kValueCacheInput];
    int32_t executed = 0;
    int32_t model_calls = 0;
    int64_t generated_token = -1;
    int64_t current_position_value = initial_position;
    int32_t finish_reason = kFinishMaxSteps;

    while (executed < max_steps) {
      const std::vector<std::shared_ptr<FlowMsg>> model_inputs = {
          current_token, current_position, current_sequence_length,
          current_key, current_slot, input_msgs[kBlockTableInput], current_value,
          input_msgs[kTilingInput]};
      std::vector<std::shared_ptr<FlowMsg>> model_outputs;
      ++model_calls;
      const auto ret = context_->RunFlowModel(
          "decode_graph_0", model_inputs, model_outputs, kRunModelTimeoutMs);
      if (ret != FLOW_FUNC_SUCCESS) {
        return fallback(kStatusModelError, executed, model_calls, generated_token);
      }
      if (model_outputs.size() != kDecoderOutputCount ||
          !IsTensor(model_outputs[0], TensorDataType::DT_FLOAT, kVocabSize) ||
          !IsTensor(model_outputs[1], TensorDataType::DT_BF16, kCacheElements) ||
          !IsTensor(model_outputs[2], TensorDataType::DT_BF16, kCacheElements) ||
          !IsTensor(model_outputs[3], TensorDataType::DT_INT64, 1)) {
        return fallback(kStatusInvalidModelOutput, executed, model_calls,
                        generated_token);
      }
      if (!ArgmaxFinite(model_outputs[0], generated_token)) {
        return fallback(kStatusNonFiniteLogits, executed, model_calls,
                        generated_token);
      }
      auto *history =
          static_cast<float *>(logits_history->GetTensor()->GetData());
      std::memcpy(history + static_cast<int64_t>(executed) * kVocabSize,
                  model_outputs[0]->GetTensor()->GetData(),
                  static_cast<size_t>(kVocabSize) * sizeof(float));
      tokens[executed] = generated_token;
      ++executed;

      int64_t next_position = -1;
      if (!ReadInt64(model_outputs[3], next_position) ||
          next_position != current_position_value + 1) {
        return fallback(kStatusPositionProgress, executed, model_calls,
                        generated_token);
      }
      current_key = model_outputs[1];
      current_value = model_outputs[2];
      current_position = model_outputs[3];
      current_position_value = next_position;

      if (generated_token == eos_token) {
        finish_reason = kFinishEos;
        break;
      }
      if (executed >= max_steps) {
        finish_reason = kFinishMaxSteps;
        break;
      }

      current_token = context_->AllocTensorMsg({1, 1}, TensorDataType::DT_INT64);
      current_sequence_length =
          context_->AllocTensorMsg({1, 1}, TensorDataType::DT_INT32);
      current_slot = context_->AllocTensorMsg({1}, TensorDataType::DT_INT32);
      if (!IsTensor(current_token, TensorDataType::DT_INT64, 1) ||
          !IsTensor(current_sequence_length, TensorDataType::DT_INT32, 1) ||
          !IsTensor(current_slot, TensorDataType::DT_INT32, 1)) {
        return fallback(kStatusAllocationFailure, executed, model_calls,
                        generated_token);
      }
      const int32_t next_slot = ComputeSlot(block_table, current_position_value);
      if (next_slot < 0) {
        return fallback(kStatusCapacityExceeded, executed, model_calls,
                        generated_token);
      }
      *static_cast<int64_t *>(current_token->GetTensor()->GetData()) =
          generated_token;
      *static_cast<int32_t *>(current_sequence_length->GetTensor()->GetData()) =
          static_cast<int32_t>(current_position_value + 1);
      *static_cast<int32_t *>(current_slot->GetTensor()->GetData()) = next_slot;
    }

    const int32_t final_sequence_length =
        static_cast<int32_t>(current_position_value + 1);
    FLOW_FUNC_LOG_INFO(
        "G4b epoch complete steps[%d] token[%ld] reason[%d] position[%ld].",
        executed, static_cast<long>(generated_token), finish_reason,
        static_cast<long>(current_position_value));
    return emit(kStatusOk, executed, finish_reason, generated_token,
                current_position_value, final_sequence_length, model_calls,
                current_key, current_value, current_position);
  }
};

REGISTER_FLOW_FUNC("g4b_resident_epoch", G4bResidentEpoch);
}  // namespace FlowFunc
