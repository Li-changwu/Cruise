#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "flow_func/flow_func_log.h"
#include "flow_func/meta_flow_func.h"

namespace FlowFunc {
namespace {
constexpr size_t kInputCount = 10;
constexpr size_t kOutputCount = 10;
constexpr size_t kDecoderOutputCount = 4;
constexpr int32_t kBatchSize = 2;
constexpr int32_t kMaxEpochSteps = 8;
constexpr int32_t kLogicalCapacity = 8;
constexpr int32_t kPhysicalBlocks = 4;
constexpr int32_t kBlocksPerRequest = 2;
constexpr int32_t kBlockSize = 128;
constexpr int32_t kVocabSize = 152064;
constexpr int32_t kRunModelTimeoutMs = 300000;
constexpr int64_t kCacheElements = 28LL * 4 * 128 * 4 * 128;
constexpr int32_t kControlInputElements = 5;
constexpr int32_t kControlOutputElements = 16;

enum InputIndex : size_t {
  kTokenInput = 0,
  kPositionInput = 1,
  kSequenceLengthInput = 2,
  kKeyCacheInput = 3,
  kSlotMappingInput = 4,
  kActiveMaskInput = 5,
  kBlockTableInput = 6,
  kValueCacheInput = 7,
  kTilingInput = 8,
  kControlInput = 9,
};

enum FinishReason : int32_t {
  kFinishNone = 0,
  kFinishEos = 1,
  kFinishMaxSteps = 2,
  kFinishFallback = 3,
  kFinishEmpty = 4,
  kFinishAlreadyFinished = 5,
};

enum Status : int32_t {
  kStatusOk = 0,
  kStatusInvalidMetadata = 201,
  kStatusCapacityExceeded = 202,
  kStatusInvalidBlockTable = 203,
  kStatusSlotMismatch = 204,
  kStatusUnsupportedSampling = 205,
  kStatusUnsupportedGraph = 206,
  kStatusModelError = 207,
  kStatusInvalidModelOutput = 208,
  kStatusNonFiniteLogits = 209,
  kStatusPositionProgress = 210,
  kStatusAllocationFailure = 211,
};

bool IsTensor(const std::shared_ptr<FlowMsg> &msg, TensorDataType dtype,
              int64_t elements) {
  if (msg == nullptr || msg->GetRetCode() != FLOW_FUNC_SUCCESS) return false;
  auto *tensor = msg->GetTensor();
  return tensor != nullptr && tensor->GetDataType() == dtype &&
         tensor->GetElementCnt() == elements && tensor->GetData() != nullptr;
}

int32_t ComputeSlot(const int32_t *block_table, int32_t request,
                    int64_t position) {
  if (request < 0 || request >= kBatchSize || position < 0 ||
      position >= kLogicalCapacity) {
    return -1;
  }
  const int32_t logical_block = static_cast<int32_t>(position / kBlockSize);
  const int32_t offset = static_cast<int32_t>(position % kBlockSize);
  if (logical_block >= kBlocksPerRequest) return -1;
  const int32_t physical_block =
      block_table[request * kBlocksPerRequest + logical_block];
  if (physical_block < 0 || physical_block >= kPhysicalBlocks) return -1;
  return physical_block * kBlockSize + offset;
}

bool ArgmaxFinite(const std::shared_ptr<FlowMsg> &msg, int32_t request,
                  int64_t &token) {
  if (!IsTensor(msg, TensorDataType::DT_FLOAT,
                static_cast<int64_t>(kBatchSize) * kVocabSize)) {
    return false;
  }
  const auto *all_logits =
      static_cast<const float *>(msg->GetTensor()->GetData());
  const auto *logits = all_logits + static_cast<int64_t>(request) * kVocabSize;
  if (!std::isfinite(logits[0])) return false;
  float best = logits[0];
  token = 0;
  for (int64_t index = 1; index < kVocabSize; ++index) {
    if (!std::isfinite(logits[index])) return false;
    if (logits[index] > best) {
      best = logits[index];
      token = index;
    }
  }
  return true;
}

int32_t CountActive(const int32_t *active) {
  int32_t count = 0;
  for (int32_t request = 0; request < kBatchSize; ++request) {
    if (active[request] != 0) ++count;
  }
  return count;
}
}  // namespace

class G4cB2ResidentEpoch : public MetaFlowFunc {
 public:
  int32_t Init() override { return FLOW_FUNC_SUCCESS; }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &inputs) override {
    if (inputs.size() != kInputCount) {
      FLOW_FUNC_LOG_ERROR("Invalid G4c B2 input count[%zu].", inputs.size());
      return FLOW_FUNC_FAILED;
    }

    auto logits_history = context_->AllocTensorMsg(
        {kMaxEpochSteps, kBatchSize, 1, kVocabSize},
        TensorDataType::DT_FLOAT);
    auto token_history = context_->AllocTensorMsg(
        {kMaxEpochSteps, kBatchSize}, TensorDataType::DT_INT64);
    auto control_output = context_->AllocTensorMsg(
        {kControlOutputElements}, TensorDataType::DT_INT32);
    if (!IsTensor(logits_history, TensorDataType::DT_FLOAT,
                  static_cast<int64_t>(kMaxEpochSteps) * kBatchSize *
                      kVocabSize) ||
        !IsTensor(token_history, TensorDataType::DT_INT64,
                  kMaxEpochSteps * kBatchSize) ||
        !IsTensor(control_output, TensorDataType::DT_INT32,
                  kControlOutputElements)) {
      FLOW_FUNC_LOG_ERROR("Failed to allocate G4c B2 history outputs.");
      return FLOW_FUNC_FAILED;
    }
    std::memset(logits_history->GetTensor()->GetData(), 0,
                logits_history->GetTensor()->GetDataSize());
    auto *history_tokens =
        static_cast<int64_t *>(token_history->GetTensor()->GetData());
    for (int32_t index = 0; index < kMaxEpochSteps * kBatchSize; ++index) {
      history_tokens[index] = -1;
    }
    auto *control =
        static_cast<int32_t *>(control_output->GetTensor()->GetData());
    std::memset(control, 0,
                static_cast<size_t>(kControlOutputElements) * sizeof(int32_t));

    if (!IsTensor(inputs[kTokenInput], TensorDataType::DT_INT64, kBatchSize) ||
        !IsTensor(inputs[kPositionInput], TensorDataType::DT_INT64,
                  kBatchSize) ||
        !IsTensor(inputs[kSequenceLengthInput], TensorDataType::DT_INT32,
                  kBatchSize) ||
        !IsTensor(inputs[kKeyCacheInput], TensorDataType::DT_BF16,
                  kCacheElements) ||
        !IsTensor(inputs[kSlotMappingInput], TensorDataType::DT_INT32,
                  kBatchSize) ||
        !IsTensor(inputs[kActiveMaskInput], TensorDataType::DT_INT32,
                  kBatchSize) ||
        !IsTensor(inputs[kBlockTableInput], TensorDataType::DT_INT32,
                  kBatchSize * kBlocksPerRequest) ||
        !IsTensor(inputs[kValueCacheInput], TensorDataType::DT_BF16,
                  kCacheElements) ||
        !IsTensor(inputs[kTilingInput], TensorDataType::DT_UINT8, 72) ||
        !IsTensor(inputs[kControlInput], TensorDataType::DT_INT32,
                  kControlInputElements)) {
      FLOW_FUNC_LOG_ERROR("Invalid G4c B2 tensor ABI.");
      return FLOW_FUNC_FAILED;
    }

    const auto *input_control = static_cast<const int32_t *>(
        inputs[kControlInput]->GetTensor()->GetData());
    const int32_t max_steps = input_control[0];
    const std::array<int32_t, kBatchSize> eos = {
        {input_control[1], input_control[2]}};
    const int32_t sampling_mode = input_control[3];
    const int32_t graph_variant = input_control[4];
    const auto *initial_token = static_cast<const int64_t *>(
        inputs[kTokenInput]->GetTensor()->GetData());
    const auto *initial_position = static_cast<const int64_t *>(
        inputs[kPositionInput]->GetTensor()->GetData());
    const auto *initial_length = static_cast<const int32_t *>(
        inputs[kSequenceLengthInput]->GetTensor()->GetData());
    const auto *initial_slot = static_cast<const int32_t *>(
        inputs[kSlotMappingInput]->GetTensor()->GetData());
    const auto *initial_active = static_cast<const int32_t *>(
        inputs[kActiveMaskInput]->GetTensor()->GetData());
    const auto *block_table = static_cast<const int32_t *>(
        inputs[kBlockTableInput]->GetTensor()->GetData());

    std::array<int32_t, kBatchSize> executed = {{0, 0}};
    std::array<int32_t, kBatchSize> finish_reason = {
        {kFinishNone, kFinishNone}};
    int32_t model_calls = 0;

    auto emit = [&](int32_t status,
                    const std::shared_ptr<FlowMsg> &final_key,
                    const std::shared_ptr<FlowMsg> &final_value,
                    const std::shared_ptr<FlowMsg> &final_token,
                    const std::shared_ptr<FlowMsg> &final_position,
                    const std::shared_ptr<FlowMsg> &final_length,
                    const std::shared_ptr<FlowMsg> &final_slot,
                    const std::shared_ptr<FlowMsg> &final_active) -> int32_t {
      const auto *final_active_values = static_cast<const int32_t *>(
          final_active->GetTensor()->GetData());
      control[0] = max_steps;
      control[1] = sampling_mode;
      control[2] = graph_variant;
      control[3] = status;
      control[4] = model_calls;
      control[5] = status == kStatusOk ? 0 : 1;
      control[6] = eos[0];
      control[7] = eos[1];
      control[8] = initial_active[0];
      control[9] = initial_active[1];
      control[10] = executed[0];
      control[11] = executed[1];
      control[12] = finish_reason[0];
      control[13] = finish_reason[1];
      control[14] = CountActive(initial_active);
      control[15] = CountActive(final_active_values);
      const std::vector<std::shared_ptr<FlowMsg>> outputs = {
          logits_history, token_history, final_key, final_value, final_token,
          final_position, final_length, final_slot, final_active,
          control_output};
      for (size_t index = 0; index < outputs.size(); ++index) {
        const auto ret = context_->SetOutput(index, outputs[index]);
        if (ret != FLOW_FUNC_SUCCESS) return ret;
      }
      return FLOW_FUNC_SUCCESS;
    };

    auto fallback = [&](int32_t status) -> int32_t {
      for (int32_t request = 0; request < kBatchSize; ++request) {
        finish_reason[request] = kFinishFallback;
      }
      FLOW_FUNC_LOG_ERROR("G4c B2 fallback status[%d] calls[%d].", status,
                          model_calls);
      return emit(status, inputs[kKeyCacheInput], inputs[kValueCacheInput],
                  inputs[kTokenInput], inputs[kPositionInput],
                  inputs[kSequenceLengthInput], inputs[kSlotMappingInput],
                  inputs[kActiveMaskInput]);
    };

    if (max_steps < 1 || max_steps > kMaxEpochSteps || sampling_mode != 0 ||
        graph_variant != 0) {
      if (sampling_mode != 0) return fallback(kStatusUnsupportedSampling);
      if (graph_variant != 0) return fallback(kStatusUnsupportedGraph);
      return fallback(kStatusInvalidMetadata);
    }
    if (eos[0] < 0 || eos[0] >= kVocabSize || eos[1] < 0 ||
        eos[1] >= kVocabSize) {
      return fallback(kStatusInvalidMetadata);
    }
    bool seen_blocks[kPhysicalBlocks] = {false, false, false, false};
    for (int32_t index = 0; index < kBatchSize * kBlocksPerRequest; ++index) {
      const int32_t block = block_table[index];
      if (block < 0 || block >= kPhysicalBlocks || seen_blocks[block]) {
        return fallback(kStatusInvalidBlockTable);
      }
      seen_blocks[block] = true;
    }
    for (int32_t request = 0; request < kBatchSize; ++request) {
      if ((initial_active[request] != 0 && initial_active[request] != 1) ||
          initial_token[request] < 0 || initial_token[request] >= kVocabSize ||
          initial_position[request] < 0 ||
          initial_position[request] >= kLogicalCapacity) {
        return fallback(kStatusInvalidMetadata);
      }
      if (initial_active[request] == 1) {
        if (initial_length[request] != initial_position[request] + 1) {
          return fallback(kStatusInvalidMetadata);
        }
        if (initial_position[request] + max_steps > kLogicalCapacity) {
          return fallback(kStatusCapacityExceeded);
        }
      } else if (initial_length[request] == 0) {
        finish_reason[request] = kFinishEmpty;
      } else if (initial_length[request] == initial_position[request] + 1) {
        finish_reason[request] = kFinishAlreadyFinished;
      } else {
        return fallback(kStatusInvalidMetadata);
      }
      if (ComputeSlot(block_table, request, initial_position[request]) !=
          initial_slot[request]) {
        return fallback(kStatusSlotMismatch);
      }
    }

    auto current_token = inputs[kTokenInput];
    auto current_position = inputs[kPositionInput];
    auto current_length = inputs[kSequenceLengthInput];
    auto current_slot = inputs[kSlotMappingInput];
    auto current_active = inputs[kActiveMaskInput];
    auto current_key = inputs[kKeyCacheInput];
    auto current_value = inputs[kValueCacheInput];
    std::array<int64_t, kBatchSize> current_position_values = {
        {initial_position[0], initial_position[1]}};

    while (model_calls < max_steps) {
      const auto *active_values = static_cast<const int32_t *>(
          current_active->GetTensor()->GetData());
      if (CountActive(active_values) == 0) break;
      const std::vector<std::shared_ptr<FlowMsg>> model_inputs = {
          current_token, current_position, current_length, current_key,
          current_slot, current_active, inputs[kBlockTableInput], current_value,
          inputs[kTilingInput]};
      std::vector<std::shared_ptr<FlowMsg>> model_outputs;
      const auto ret = context_->RunFlowModel(
          "decode_graph_0", model_inputs, model_outputs, kRunModelTimeoutMs);
      ++model_calls;
      if (ret != FLOW_FUNC_SUCCESS) return fallback(kStatusModelError);
      if (model_outputs.size() != kDecoderOutputCount ||
          !IsTensor(model_outputs[0], TensorDataType::DT_FLOAT,
                    static_cast<int64_t>(kBatchSize) * kVocabSize) ||
          !IsTensor(model_outputs[1], TensorDataType::DT_BF16,
                    kCacheElements) ||
          !IsTensor(model_outputs[2], TensorDataType::DT_BF16,
                    kCacheElements) ||
          !IsTensor(model_outputs[3], TensorDataType::DT_INT64, kBatchSize)) {
        return fallback(kStatusInvalidModelOutput);
      }

      auto next_token = context_->AllocTensorMsg({kBatchSize, 1},
                                                  TensorDataType::DT_INT64);
      auto next_length = context_->AllocTensorMsg({kBatchSize, 1},
                                                   TensorDataType::DT_INT32);
      auto next_slot = context_->AllocTensorMsg({kBatchSize},
                                                 TensorDataType::DT_INT32);
      auto next_active = context_->AllocTensorMsg({kBatchSize},
                                                   TensorDataType::DT_INT32);
      if (!IsTensor(next_token, TensorDataType::DT_INT64, kBatchSize) ||
          !IsTensor(next_length, TensorDataType::DT_INT32, kBatchSize) ||
          !IsTensor(next_slot, TensorDataType::DT_INT32, kBatchSize) ||
          !IsTensor(next_active, TensorDataType::DT_INT32, kBatchSize)) {
        return fallback(kStatusAllocationFailure);
      }
      std::memcpy(next_token->GetTensor()->GetData(),
                  current_token->GetTensor()->GetData(),
                  static_cast<size_t>(kBatchSize) * sizeof(int64_t));
      std::memcpy(next_length->GetTensor()->GetData(),
                  current_length->GetTensor()->GetData(),
                  static_cast<size_t>(kBatchSize) * sizeof(int32_t));
      std::memcpy(next_slot->GetTensor()->GetData(),
                  current_slot->GetTensor()->GetData(),
                  static_cast<size_t>(kBatchSize) * sizeof(int32_t));
      std::memcpy(next_active->GetTensor()->GetData(),
                  current_active->GetTensor()->GetData(),
                  static_cast<size_t>(kBatchSize) * sizeof(int32_t));

      auto *next_token_values =
          static_cast<int64_t *>(next_token->GetTensor()->GetData());
      auto *next_length_values =
          static_cast<int32_t *>(next_length->GetTensor()->GetData());
      auto *next_slot_values =
          static_cast<int32_t *>(next_slot->GetTensor()->GetData());
      auto *next_active_values =
          static_cast<int32_t *>(next_active->GetTensor()->GetData());
      const auto *next_position_values = static_cast<const int64_t *>(
          model_outputs[3]->GetTensor()->GetData());
      const auto *step_logits =
          static_cast<const float *>(model_outputs[0]->GetTensor()->GetData());
      auto *all_history =
          static_cast<float *>(logits_history->GetTensor()->GetData());

      for (int32_t request = 0; request < kBatchSize; ++request) {
        if (active_values[request] == 0) {
          if (next_position_values[request] !=
              current_position_values[request]) {
            return fallback(kStatusPositionProgress);
          }
          continue;
        }
        int64_t generated_token = -1;
        if (!ArgmaxFinite(model_outputs[0], request, generated_token)) {
          return fallback(kStatusNonFiniteLogits);
        }
        if (next_position_values[request] !=
            current_position_values[request] + 1) {
          return fallback(kStatusPositionProgress);
        }
        const int64_t history_offset =
            (static_cast<int64_t>(model_calls - 1) * kBatchSize + request) *
            kVocabSize;
        const int64_t logits_offset =
            static_cast<int64_t>(request) * kVocabSize;
        std::memcpy(all_history + history_offset, step_logits + logits_offset,
                    static_cast<size_t>(kVocabSize) * sizeof(float));
        history_tokens[(model_calls - 1) * kBatchSize + request] =
            generated_token;
        ++executed[request];
        current_position_values[request] = next_position_values[request];
        next_token_values[request] = generated_token;
        next_length_values[request] =
            static_cast<int32_t>(current_position_values[request] + 1);
        if (current_position_values[request] == kLogicalCapacity) {
          next_slot_values[request] = -1;
        } else {
          const int32_t slot = ComputeSlot(block_table, request,
                                           current_position_values[request]);
          if (slot < 0) return fallback(kStatusCapacityExceeded);
          next_slot_values[request] = slot;
        }
        if (generated_token == eos[request]) {
          next_active_values[request] = 0;
          finish_reason[request] = kFinishEos;
        }
      }

      current_key = model_outputs[1];
      current_value = model_outputs[2];
      current_position = model_outputs[3];
      current_token = next_token;
      current_length = next_length;
      current_slot = next_slot;
      current_active = next_active;
    }

    for (int32_t request = 0; request < kBatchSize; ++request) {
      if (initial_active[request] == 1 &&
          finish_reason[request] == kFinishNone) {
        finish_reason[request] = kFinishMaxSteps;
      }
    }
    FLOW_FUNC_LOG_INFO(
        "G4c B2 epoch complete calls[%d] executed[%d,%d] reason[%d,%d].",
        model_calls, executed[0], executed[1], finish_reason[0],
        finish_reason[1]);
    return emit(kStatusOk, current_key, current_value, current_token,
                current_position, current_length, current_slot, current_active);
  }
};

REGISTER_FLOW_FUNC("g4c_b2_resident_epoch", G4cB2ResidentEpoch);
}  // namespace FlowFunc
