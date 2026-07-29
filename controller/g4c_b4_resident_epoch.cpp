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
constexpr size_t kInputCount = 8;
constexpr size_t kOutputCount = 2;
constexpr size_t kDecoderOutputCount = 4;
constexpr int32_t kBatchSize = 4;
constexpr int32_t kMaxEpochSteps = 8;
constexpr int32_t kLogicalCapacity = 8;
constexpr int32_t kPhysicalBlocks = 8;
constexpr int32_t kBlocksPerRequest = 2;
constexpr int32_t kBlockSize = 128;
constexpr int32_t kVocabSize = 152064;
constexpr int32_t kRunModelTimeoutMs = 300000;
constexpr int64_t kCacheElements = 28LL * 8 * 128 * 4 * 128;
constexpr int64_t kImportCacheElements = 28LL * 4 * 128 * 4 * 128;
constexpr int64_t kImportPayloadElements = 2 * kImportCacheElements * 2;
constexpr int32_t kImportGraphFlag = 0x100;
constexpr int32_t kControlInputElements = 1 + kBatchSize + 2 + kBatchSize;
constexpr int32_t kControlOutputElements = 6 + 4 * kBatchSize + 2 + kBatchSize;
constexpr int32_t kControlGenerationInputOffset = 3 + kBatchSize;
constexpr int32_t kControlEosOffset = 6;
constexpr int32_t kControlInitialActiveOffset = kControlEosOffset + kBatchSize;
constexpr int32_t kControlExecutedOffset =
    kControlInitialActiveOffset + kBatchSize;
constexpr int32_t kControlReasonOffset = kControlExecutedOffset + kBatchSize;
constexpr int32_t kControlInitialCount = kControlReasonOffset + kBatchSize;
constexpr int32_t kControlFinalCount = kControlInitialCount + 1;
constexpr int32_t kControlGenerationOutputOffset = kControlFinalCount + 1;

enum InputIndex : size_t {
  kTokenInput = 0,
  kPositionInput = 1,
  kSequenceLengthInput = 2,
  kSlotMappingInput = 3,
  kActiveMaskInput = 4,
  kBlockTableInput = 5,
  kTilingInput = 6,
  kControlInput = 7,
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
  kStatusResidentContinuity = 212,
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

bool ClearCacheRow(const std::shared_ptr<FlowMsg> &cache, int32_t row) {
  if (!IsTensor(cache, TensorDataType::DT_BF16, kCacheElements) || row < 0 ||
      row >= kBatchSize) {
    return false;
  }
  constexpr size_t kBlockBytes =
      128ULL * 4ULL * 128ULL * sizeof(uint16_t);
  auto *data = static_cast<uint8_t *>(cache->GetTensor()->GetData());
  for (int32_t layer = 0; layer < 28; ++layer) {
    for (int32_t local_block = 0; local_block < kBlocksPerRequest;
         ++local_block) {
      const int32_t block = row * kBlocksPerRequest + local_block;
      const size_t offset =
          (static_cast<size_t>(layer) * kPhysicalBlocks + block) * kBlockBytes;
      std::memset(data + offset, 0, kBlockBytes);
    }
  }
  return true;
}

void Adler32Update(const uint8_t *data, size_t bytes,
                   uint32_t &first, uint32_t &second) {
  constexpr uint32_t kModulus = 65521;
  while (bytes > 0) {
    const size_t chunk = bytes > 5552 ? 5552 : bytes;
    for (size_t index = 0; index < chunk; ++index) {
      first += data[index];
      second += first;
    }
    first %= kModulus;
    second %= kModulus;
    data += chunk;
    bytes -= chunk;
  }
}
}  // namespace

class G4cB4ResidentEpoch : public MetaFlowFunc {
 public:
  int32_t Init() override {
    resident_generation_.fill(0);
    resident_token_.fill(0);
    resident_position_.fill(0);
    resident_length_.fill(0);
    resident_valid_.fill(0);
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &inputs) override {
    if (inputs.size() != kInputCount) {
      FLOW_FUNC_LOG_ERROR("Invalid G4c B4 input count[%zu].", inputs.size());
      return FLOW_FUNC_FAILED;
    }

    auto token_history = context_->AllocTensorMsg(
        {kMaxEpochSteps, kBatchSize}, TensorDataType::DT_INT64);
    auto control_output = context_->AllocTensorMsg(
        {kControlOutputElements}, TensorDataType::DT_INT32);
    if (!IsTensor(token_history, TensorDataType::DT_INT64,
                  kMaxEpochSteps * kBatchSize) ||
        !IsTensor(control_output, TensorDataType::DT_INT32,
                  kControlOutputElements)) {
      FLOW_FUNC_LOG_ERROR("Failed to allocate G4c B4 history outputs.");
      return FLOW_FUNC_FAILED;
    }
    auto *history_tokens =
        static_cast<int64_t *>(token_history->GetTensor()->GetData());
    for (int32_t index = 0; index < kMaxEpochSteps * kBatchSize; ++index) {
      history_tokens[index] = -1;
    }
    auto *control =
        static_cast<int32_t *>(control_output->GetTensor()->GetData());
    std::memset(control, 0,
                static_cast<size_t>(kControlOutputElements) * sizeof(int32_t));

    if (!IsTensor(inputs[kControlInput], TensorDataType::DT_INT32,
                  kControlInputElements)) {
      FLOW_FUNC_LOG_ERROR("Invalid G4c B4 control ABI.");
      return FLOW_FUNC_FAILED;
    }
    const auto *input_control = static_cast<const int32_t *>(
        inputs[kControlInput]->GetTensor()->GetData());
    const int32_t encoded_graph_variant = input_control[2 + kBatchSize];
    const bool importing = (encoded_graph_variant & kImportGraphFlag) != 0;
    const int32_t import_mask =
        importing ? encoded_graph_variant & (kImportGraphFlag - 1) : 0;
    const int32_t graph_variant =
        importing ? encoded_graph_variant & ~(kImportGraphFlag | 0xF) :
                    encoded_graph_variant;

    auto token_input = inputs[importing ? 1 : kTokenInput];
    auto position_input = inputs[importing ? 2 : kPositionInput];
    auto sequence_length_input =
        inputs[importing ? 3 : kSequenceLengthInput];
    auto active_mask_input = inputs[importing ? 4 : kActiveMaskInput];
    auto block_table_input = inputs[importing ? 5 : kBlockTableInput];
    auto tiling_input = inputs[importing ? 6 : kTilingInput];
    auto slot_mapping_input = importing
                                  ? context_->AllocTensorMsg(
                                        {kBatchSize}, TensorDataType::DT_INT32)
                                  : inputs[kSlotMappingInput];
    const bool common_inputs_valid =
        IsTensor(token_input, TensorDataType::DT_INT64, kBatchSize) &&
        IsTensor(position_input, TensorDataType::DT_INT64, kBatchSize) &&
        IsTensor(sequence_length_input, TensorDataType::DT_INT32, kBatchSize) &&
        IsTensor(slot_mapping_input, TensorDataType::DT_INT32, kBatchSize) &&
        IsTensor(active_mask_input, TensorDataType::DT_INT32, kBatchSize) &&
        IsTensor(block_table_input, TensorDataType::DT_INT32,
                 kBatchSize * kBlocksPerRequest) &&
        IsTensor(tiling_input, TensorDataType::DT_UINT8, 72);
    if (!common_inputs_valid ||
        (importing &&
         !IsTensor(inputs[0], TensorDataType::DT_UINT8,
                   kImportPayloadElements))) {
      FLOW_FUNC_LOG_ERROR("Invalid G4c B4 tensor ABI.");
      return FLOW_FUNC_FAILED;
    }

    if (resident_key_ == nullptr || resident_value_ == nullptr) {
      resident_key_ = context_->AllocTensorMsg(
          {28, 8, 128, 4, 128}, TensorDataType::DT_BF16);
      resident_value_ = context_->AllocTensorMsg(
          {28, 8, 128, 4, 128}, TensorDataType::DT_BF16);
      if (!IsTensor(resident_key_, TensorDataType::DT_BF16, kCacheElements) ||
          !IsTensor(resident_value_, TensorDataType::DT_BF16,
                    kCacheElements)) {
        FLOW_FUNC_LOG_ERROR("Failed to allocate resident Paged-KV state.");
        return FLOW_FUNC_FAILED;
      }
      std::memset(resident_key_->GetTensor()->GetData(), 0,
                  resident_key_->GetTensor()->GetDataSize());
      std::memset(resident_value_->GetTensor()->GetData(), 0,
                  resident_value_->GetTensor()->GetDataSize());
    }

    const int32_t max_steps = input_control[0];
    std::array<int32_t, kBatchSize> eos{};
    for (int32_t request = 0; request < kBatchSize; ++request) {
      eos[request] = input_control[1 + request];
    }
    const int32_t sampling_mode = input_control[1 + kBatchSize];
    const auto *row_generation =
        input_control + kControlGenerationInputOffset;
    const auto *initial_token = static_cast<const int64_t *>(
        token_input->GetTensor()->GetData());
    const auto *initial_position = static_cast<const int64_t *>(
        position_input->GetTensor()->GetData());
    const auto *initial_length = static_cast<const int32_t *>(
        sequence_length_input->GetTensor()->GetData());
    auto *initial_slot = static_cast<int32_t *>(
        slot_mapping_input->GetTensor()->GetData());
    const auto *initial_active = static_cast<const int32_t *>(
        active_mask_input->GetTensor()->GetData());
    const auto *block_table = static_cast<const int32_t *>(
        block_table_input->GetTensor()->GetData());
    if (importing) {
      for (int32_t request = 0; request < kBatchSize; ++request) {
        initial_slot[request] =
            ComputeSlot(block_table, request, initial_position[request]);
      }
    }

    std::array<int32_t, kBatchSize> executed{};
    std::array<int32_t, kBatchSize> finish_reason{};
    int32_t model_calls = 0;
    uint32_t import_checksum = 0;

    auto emit = [&](int32_t status,
                    const std::shared_ptr<FlowMsg> &final_active) -> int32_t {
      const auto *final_active_values = static_cast<const int32_t *>(
          final_active->GetTensor()->GetData());
      control[0] = max_steps;
      control[1] = sampling_mode;
      control[2] = graph_variant;
      control[3] = status;
      control[4] = model_calls;
      control[5] = importing ? static_cast<int32_t>(import_checksum) :
                               (status == kStatusOk ? 0 : 1);
      for (int32_t request = 0; request < kBatchSize; ++request) {
        control[kControlEosOffset + request] = eos[request];
        control[kControlInitialActiveOffset + request] = initial_active[request];
        control[kControlExecutedOffset + request] = executed[request];
        control[kControlReasonOffset + request] = finish_reason[request];
        control[kControlGenerationOutputOffset + request] =
            initial_active[request] == 1 ? row_generation[request] : 0;
      }
      control[kControlInitialCount] = CountActive(initial_active);
      control[kControlFinalCount] = CountActive(final_active_values);
      const std::vector<std::shared_ptr<FlowMsg>> outputs = {
          token_history, control_output};
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
      FLOW_FUNC_LOG_ERROR("G4c B4 fallback status[%d] calls[%d].", status,
                          model_calls);
      return emit(status, active_mask_input);
    };

    if (max_steps < 1 || max_steps > kMaxEpochSteps || sampling_mode != 0 ||
        graph_variant != 0) {
      if (sampling_mode != 0) return fallback(kStatusUnsupportedSampling);
      if (graph_variant != 0) return fallback(kStatusUnsupportedGraph);
      return fallback(kStatusInvalidMetadata);
    }
    for (int32_t request = 0; request < kBatchSize; ++request) {
      if (eos[request] < 0 || eos[request] >= kVocabSize) {
        return fallback(kStatusInvalidMetadata);
      }
    }
    bool seen_blocks[kPhysicalBlocks] = {};
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
        if (row_generation[request] <= 0) {
          return fallback(kStatusInvalidMetadata);
        }
        if (initial_length[request] != initial_position[request] + 1) {
          return fallback(kStatusInvalidMetadata);
        }
        if (initial_position[request] + max_steps > kLogicalCapacity) {
          return fallback(kStatusCapacityExceeded);
        }
      } else if (initial_length[request] == 0) {
        if (row_generation[request] != 0) {
          return fallback(kStatusInvalidMetadata);
        }
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

    if (importing) {
      if (import_mask <= 0 || import_mask >= (1 << kBatchSize)) {
        return fallback(kStatusInvalidMetadata);
      }
      constexpr size_t kBlockBytes =
          kBlockSize * 4ULL * 128ULL * sizeof(uint16_t);
      const auto *payload = static_cast<const uint8_t *>(
          inputs[0]->GetTensor()->GetData());
      const auto *import_key = payload;
      const auto *import_value =
          payload + kImportCacheElements * sizeof(uint16_t);
      auto *resident_key = static_cast<uint8_t *>(
          resident_key_->GetTensor()->GetData());
      auto *resident_value = static_cast<uint8_t *>(
          resident_value_->GetTensor()->GetData());
      for (int32_t request = 0; request < kBatchSize; ++request) {
        if ((import_mask & (1 << request)) == 0) continue;
        if (initial_active[request] != 1 || row_generation[request] <= 0) {
          return fallback(kStatusInvalidMetadata);
        }
        if (!ClearCacheRow(resident_key_, request) ||
            !ClearCacheRow(resident_value_, request)) {
          return fallback(kStatusAllocationFailure);
        }
        for (int32_t layer = 0; layer < 28; ++layer) {
          const size_t source_offset =
              (static_cast<size_t>(layer) * kBatchSize + request) *
              kBlockBytes;
          const size_t destination_offset =
              (static_cast<size_t>(layer) * kPhysicalBlocks +
               request * kBlocksPerRequest) *
              kBlockBytes;
          std::memcpy(resident_key + destination_offset,
                      import_key + source_offset, kBlockBytes);
          std::memcpy(resident_value + destination_offset,
                      import_value + source_offset, kBlockBytes);
        }
        resident_generation_[request] = row_generation[request];
        resident_token_[request] = initial_token[request];
        resident_position_[request] = initial_position[request];
        resident_length_[request] = initial_length[request];
        resident_valid_[request] = 1;
      }
      uint32_t checksum_first = 1;
      uint32_t checksum_second = 0;
      for (auto *cache : {resident_key, resident_value}) {
        for (int32_t layer = 0; layer < 28; ++layer) {
          for (int32_t request = 0; request < kBatchSize; ++request) {
            if ((import_mask & (1 << request)) == 0) continue;
            const size_t destination_offset =
                (static_cast<size_t>(layer) * kPhysicalBlocks +
                 request * kBlocksPerRequest) *
                kBlockBytes;
            Adler32Update(cache + destination_offset, kBlockBytes,
                          checksum_first, checksum_second);
          }
        }
      }
      import_checksum = (checksum_second << 16) | checksum_first;
    }

    std::array<int32_t, kBatchSize> new_generation{};
    for (int32_t request = 0; request < kBatchSize; ++request) {
      if (initial_active[request] == 0) continue;
      if (resident_generation_[request] == row_generation[request]) {
        if (resident_valid_[request] == 0 ||
            resident_token_[request] != initial_token[request] ||
            resident_position_[request] != initial_position[request] ||
            resident_length_[request] != initial_length[request]) {
          return fallback(kStatusResidentContinuity);
        }
      } else {
        if (initial_position[request] != 0 || initial_length[request] != 1) {
          return fallback(kStatusResidentContinuity);
        }
        new_generation[request] = 1;
      }
    }
    for (int32_t request = 0; request < kBatchSize; ++request) {
      if (new_generation[request] == 0) continue;
      if (!ClearCacheRow(resident_key_, request) ||
          !ClearCacheRow(resident_value_, request)) {
        return fallback(kStatusAllocationFailure);
      }
    }

    auto current_token = token_input;
    auto current_position = position_input;
    auto current_length = sequence_length_input;
    auto current_slot = slot_mapping_input;
    auto current_active = active_mask_input;
    auto current_key = resident_key_;
    auto current_value = resident_value_;
    std::array<int64_t, kBatchSize> current_position_values{};
    for (int32_t request = 0; request < kBatchSize; ++request) {
      current_position_values[request] = initial_position[request];
    }

    while (model_calls < max_steps) {
      const auto *active_values = static_cast<const int32_t *>(
          current_active->GetTensor()->GetData());
      if (CountActive(active_values) == 0) break;
      const std::vector<std::shared_ptr<FlowMsg>> model_inputs = {
          current_token, current_position, current_length, current_key,
          current_slot, current_active, block_table_input, current_value,
          tiling_input};
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

    resident_key_ = current_key;
    resident_value_ = current_value;
    const auto *final_tokens = static_cast<const int64_t *>(
        current_token->GetTensor()->GetData());
    const auto *final_positions = static_cast<const int64_t *>(
        current_position->GetTensor()->GetData());
    const auto *final_lengths = static_cast<const int32_t *>(
        current_length->GetTensor()->GetData());
    for (int32_t request = 0; request < kBatchSize; ++request) {
      if (initial_active[request] == 0) continue;
      resident_generation_[request] = row_generation[request];
      resident_token_[request] = final_tokens[request];
      resident_position_[request] = final_positions[request];
      resident_length_[request] = final_lengths[request];
      resident_valid_[request] = 1;
    }

    FLOW_FUNC_LOG_INFO("G4c B4 epoch complete calls[%d] initial_active[%d] "
                       "final_active[%d].",
                       model_calls, CountActive(initial_active),
                       CountActive(static_cast<const int32_t *>(
                           current_active->GetTensor()->GetData())));
    return emit(kStatusOk, current_active);
  }

 private:
  std::shared_ptr<FlowMsg> resident_key_;
  std::shared_ptr<FlowMsg> resident_value_;
  std::array<int32_t, kBatchSize> resident_generation_{};
  std::array<int64_t, kBatchSize> resident_token_{};
  std::array<int64_t, kBatchSize> resident_position_{};
  std::array<int32_t, kBatchSize> resident_length_{};
  std::array<int32_t, kBatchSize> resident_valid_{};
};

REGISTER_FLOW_FUNC("g4c_b4_resident_epoch", G4cB4ResidentEpoch);
}  // namespace FlowFunc
