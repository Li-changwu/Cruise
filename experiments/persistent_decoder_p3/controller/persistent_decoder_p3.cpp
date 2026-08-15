#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "flow_func/flow_func_log.h"
#include "flow_func/meta_multi_func.h"

namespace FlowFunc {
namespace {
constexpr size_t kRows = 4;
constexpr size_t kLayers = 28;
constexpr size_t kKvHeads = 4;
constexpr size_t kHeadDim = 128;
constexpr size_t kBlockSize = 128;
constexpr size_t kBlocksPerRequest = 3;
constexpr size_t kPhysicalBlocks = kRows * kBlocksPerRequest;
constexpr size_t kLogicalCapacity = 384;
constexpr size_t kMaxPromptTokens = 128;
constexpr size_t kMaxTargetSteps = 256;
constexpr size_t kEventHeaderElements = 16;
constexpr size_t kEventElements = kEventHeaderElements + kMaxPromptTokens;
constexpr size_t kOutputElements = 20;
constexpr int32_t kRunModelTimeoutMs = 300000;
constexpr int64_t kEventAdmit = 1;
constexpr int64_t kEventCredit = 2;
constexpr int64_t kEventCancel = 3;
constexpr int64_t kEventShutdown = 4;
constexpr int64_t kOutputAdmitAck = 1;
constexpr int64_t kOutputCommit = 2;
constexpr int64_t kOutputRetireComplete = 3;
constexpr int64_t kOutputRetireCancelled = 4;
constexpr int64_t kOutputCreditAck = 5;
constexpr int64_t kOutputQuiescent = 6;
constexpr int64_t kOutputCumulativeAck = 7;
constexpr int64_t kOutputRejected = 8;
constexpr int64_t kOutputShutdown = 9;
constexpr int64_t kFlagIgnoreEos = 1;
constexpr int64_t kNoToken = -1;

enum Status : int64_t {
  kStatusOk = 0,
  kStatusWrongOwner = 1,
  kStatusStaleEvent = 2,
  kStatusNoCapacity = 3,
  kStatusBadGeneration = 4,
  kStatusBadRow = 5,
  kStatusBadPrompt = 6,
  kStatusBadTarget = 7,
  kStatusBadCredit = 8,
  kStatusStaleCancel = 9,
  kStatusNotActive = 10,
  kStatusUnknownEvent = 11,
  kStatusBadMessage = 12,
  kStatusDuplicate = 13,
  kStatusBusy = 14,
  kStatusModelError = 15,
  kStatusInvalidModelOutput = 16,
  kStatusPositionProgress = 17,
  kStatusCapacityExceeded = 18,
  kStatusAllocationFailure = 19,
  kStatusBadCohort = 20,
};

struct Event {
  int64_t owner = 0;
  int64_t type = 0;
  int64_t seq = 0;
  int64_t request = 0;
  int64_t generation = 0;
  int64_t row = -1;
  int64_t prompt_len = 0;
  int64_t target_steps = 0;
  int64_t credit = 0;
  int64_t cancel_seq = 0;
  int64_t eos_token = 151645;
  int64_t flags = 0;
  int64_t reserved0 = 0;
  int64_t reserved1 = 0;
  int64_t reserved2 = 0;
  int64_t reserved3 = 0;
  std::array<int64_t, kMaxPromptTokens> prompt{};
};

struct Row {
  bool active = false;
  bool staged = false;
  int64_t request = 0;
  int64_t generation = 0;
  int64_t last_generation = 0;
  int64_t prompt_len = 0;
  int64_t prompt_index = 0;
  int64_t target_steps = 0;
  int64_t credit = 0;
  int64_t commits = 0;
  int64_t token = 0;
  int64_t position = 0;
  int64_t cancel_seq = 0;
  int64_t eos_token = 151645;
  int64_t flags = 0;
  int64_t cohort_id = 0;
  int64_t cohort_size = 0;
  std::array<int64_t, kMaxPromptTokens> prompt{};
};

bool IsTensor(const std::shared_ptr<FlowMsg> &message, TensorDataType type,
              int64_t elements) {
  if (message == nullptr || message->GetRetCode() != FLOW_FUNC_SUCCESS ||
      message->GetMsgType() != MsgType::MSG_TYPE_TENSOR_DATA) {
    return false;
  }
  const auto *tensor = message->GetTensor();
  return tensor != nullptr && tensor->GetDataType() == type &&
         tensor->GetElementCnt() == elements && tensor->GetData() != nullptr;
}

bool SameEvent(const Event &left, const Event &right) {
  if (left.owner != right.owner || left.type != right.type ||
      left.seq != right.seq || left.request != right.request ||
      left.generation != right.generation || left.row != right.row ||
      left.prompt_len != right.prompt_len ||
      left.target_steps != right.target_steps || left.credit != right.credit ||
      left.cancel_seq != right.cancel_seq || left.eos_token != right.eos_token ||
      left.flags != right.flags || left.reserved0 != right.reserved0 ||
      left.reserved1 != right.reserved1 || left.reserved2 != right.reserved2 ||
      left.reserved3 != right.reserved3) {
    return false;
  }
  return left.prompt == right.prompt;
}

bool ReadEvent(const std::shared_ptr<FlowMsg> &message, Event &event) {
  if (!IsTensor(message, TensorDataType::DT_INT64, kEventElements) ||
      message->GetTensor()->GetDataSize() != kEventElements * sizeof(int64_t)) {
    return false;
  }
  const auto *values =
      static_cast<const int64_t *>(message->GetTensor()->GetData());
  event.owner = values[0];
  event.type = values[1];
  event.seq = values[2];
  event.request = values[3];
  event.generation = values[4];
  event.row = values[5];
  event.prompt_len = values[6];
  event.target_steps = values[7];
  event.credit = values[8];
  event.cancel_seq = values[9];
  event.eos_token = values[10];
  event.flags = values[11];
  event.reserved0 = values[12];
  event.reserved1 = values[13];
  event.reserved2 = values[14];
  event.reserved3 = values[15];
  std::memcpy(event.prompt.data(), values + kEventHeaderElements,
              kMaxPromptTokens * sizeof(int64_t));
  return true;
}

int32_t ComputeSlot(int32_t row, int64_t position) {
  if (row < 0 || row >= static_cast<int32_t>(kRows) || position < 0 ||
      position >= static_cast<int64_t>(kLogicalCapacity)) {
    return -1;
  }
  const int32_t block = row * static_cast<int32_t>(kBlocksPerRequest) +
                        static_cast<int32_t>(position / kBlockSize);
  return block * static_cast<int32_t>(kBlockSize) +
         static_cast<int32_t>(position % kBlockSize);
}

int64_t PageCursor(int64_t position) {
  return position < 0 ? -1 : position / static_cast<int64_t>(kBlockSize);
}

void Adler32Update(const uint8_t *data, size_t bytes, uint32_t &first,
                   uint32_t &second) {
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

class PersistentDecoderP3 : public MetaMultiFunc {
 public:
  int32_t Init(const std::shared_ptr<MetaParams> &params) override {
    if (params == nullptr || params->GetInputNum() != 1 ||
        params->GetOutputNum() != 1 ||
        params->GetAttr("owner_instance_id", owner_) != FLOW_FUNC_SUCCESS ||
        owner_ <= 0) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    ResetState();
    return FLOW_FUNC_SUCCESS;
  }

  int32_t ResetFlowFuncState(const std::shared_ptr<MetaParams> &params) override {
    (void)params;
    ResetState();
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::shared_ptr<MetaRunContext> &context,
               const std::vector<std::shared_ptr<FlowMsgQueue>> &queues) {
    if (context == nullptr || queues.size() != 1 || queues[0] == nullptr) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    while (true) {
      std::shared_ptr<FlowMsg> message;
      const int32_t timeout = HasEligibleRow() ? 0 : -1;
      const int32_t dequeue = queues[0]->Dequeue(message, timeout);
      if (dequeue == FLOW_FUNC_SUCCESS) {
        Event event;
        if (!ReadEvent(message, event)) {
          if (Emit(context, kOutputRejected, last_event_seq_, 0, 0, -1, 0,
                   kNoToken, 0, 0, 0, 0, 0, kStatusBadMessage, 0) !=
              FLOW_FUNC_SUCCESS) {
            return FLOW_FUNC_FAILED;
          }
          continue;
        }
        bool shutdown = false;
        const int32_t handled = HandleEvent(context, event, shutdown);
        if (handled != FLOW_FUNC_SUCCESS) return handled;
        if (shutdown) return FLOW_FUNC_SUCCESS;
        continue;
      }
      if (dequeue != FLOW_FUNC_ERR_TIME_OUT_ERROR) return dequeue;
      if (HasEligibleRow() && RunQuantum(context) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
      }
    }
  }

 private:
  void ResetState() {
    rows_.fill(Row{});
    last_event_ = Event{};
    last_event_seq_ = 0;
    has_last_event_ = false;
    last_event_row_ = -1;
    total_events_ = 0;
    total_duplicates_ = 0;
    total_rejected_ = 0;
    total_admitted_ = 0;
    total_retired_ = 0;
    total_commits_ = 0;
    aicore_calls_ = 0;
    next_commit_seq_ = 0;
    state_key_.reset();
    state_value_.reset();
  }

  bool ValidRow(int64_t row) const {
    return row >= 0 && row < static_cast<int64_t>(kRows);
  }

  bool HasActiveRow() const {
    for (const auto &row : rows_) {
      if (row.active) return true;
    }
    return false;
  }

  bool Eligible(size_t index) const {
    const auto &row = rows_[index];
    if (!row.active || row.staged) return false;
    const bool ingesting = row.prompt_index < row.prompt_len;
    if (!ingesting) return row.credit > 0;
    const bool final_prompt_step = ingesting &&
                                   row.prompt_index + 1 == row.prompt_len;
    return !final_prompt_step || row.credit > 0;
  }

  bool HasEligibleRow() const {
    for (size_t index = 0; index < kRows; ++index) {
      if (Eligible(index)) return true;
    }
    return false;
  }

  uint32_t RowChecksum(int32_t row) const {
    if (!IsTensor(state_key_, TensorDataType::DT_BF16,
                  static_cast<int64_t>(kLayers * kPhysicalBlocks * kBlockSize *
                                       kKvHeads * kHeadDim)) ||
        !IsTensor(state_value_, TensorDataType::DT_BF16,
                  static_cast<int64_t>(kLayers * kPhysicalBlocks * kBlockSize *
                                       kKvHeads * kHeadDim)) ||
        row < 0 || row >= static_cast<int32_t>(kRows)) {
      return 0;
    }
    constexpr size_t kBlockBytes = kBlockSize * kKvHeads * kHeadDim * 2;
    const auto *key = static_cast<const uint8_t *>(
        state_key_->GetTensor()->GetData());
    const auto *value = static_cast<const uint8_t *>(
        state_value_->GetTensor()->GetData());
    uint32_t first = 1;
    uint32_t second = 0;
    for (const auto *cache : {key, value}) {
      for (size_t layer = 0; layer < kLayers; ++layer) {
        for (size_t page = 0; page < kBlocksPerRequest; ++page) {
          const size_t physical = static_cast<size_t>(row) * kBlocksPerRequest + page;
          const size_t offset = (layer * kPhysicalBlocks + physical) * kBlockBytes;
          Adler32Update(cache + offset, kBlockBytes, first, second);
        }
      }
    }
    return (second << 16) | first;
  }

  int32_t Emit(const std::shared_ptr<MetaRunContext> &context,
               int64_t output_type, int64_t event_seq, int64_t request,
               int64_t generation, int64_t row, int64_t commit_seq,
               int64_t token, int64_t position, int64_t remaining,
               int64_t credit, int64_t cancel_seq, int64_t status,
               int64_t finish_reason, bool eos = false) {
    auto output = context->AllocTensorMsg(
        {static_cast<int64_t>(kOutputElements)}, TensorDataType::DT_INT64);
    if (!IsTensor(output, TensorDataType::DT_INT64, kOutputElements)) {
      return FLOW_FUNC_FAILED;
    }
    const int64_t page = ValidRow(row) ? PageCursor(position) : -1;
    const bool final_state = output_type == kOutputRetireComplete ||
                             output_type == kOutputRetireCancelled ||
                             output_type == kOutputShutdown;
    const int64_t checksum =
        final_state && ValidRow(row) ? RowChecksum(static_cast<int32_t>(row)) : 0;
    const int64_t values[kOutputElements] = {
        owner_, output_type, event_seq, request, generation, row,
        commit_seq, token, position, page, remaining, credit, cancel_seq,
        status, aicore_calls_, total_commits_, total_retired_, checksum,
        finish_reason, quiescent_count_};
    std::memcpy(output->GetTensor()->GetData(), values, sizeof(values));
    output->SetTransactionId(static_cast<uint64_t>(event_seq));
    if (eos) output->SetFlowFlags(static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS));
    return context->SetOutput(0, output);
  }

  int32_t EmitRow(const std::shared_ptr<MetaRunContext> &context,
                  int64_t output_type, int64_t event_seq, int32_t row,
                  int64_t token, int64_t cancel_seq, int64_t status,
                  int64_t finish_reason = 0) {
    if (!ValidRow(row)) {
      return Emit(context, output_type, event_seq, 0, 0, -1, 0, token, 0, 0,
                  0, cancel_seq, status, finish_reason);
    }
    const auto &state = rows_[static_cast<size_t>(row)];
    return Emit(context, output_type, event_seq, state.request,
                state.generation, row, state.commits, token, state.position,
                state.target_steps - state.commits, state.credit, cancel_seq,
                status, finish_reason);
  }

  int32_t EmitDuplicate(const std::shared_ptr<MetaRunContext> &context,
                        const Event &event) {
    ++total_duplicates_;
    return EmitRow(context, kOutputCumulativeAck, event.seq,
                   last_event_row_, kNoToken, event.cancel_seq,
                   kStatusDuplicate);
  }

  int32_t Reject(const std::shared_ptr<MetaRunContext> &context,
                 const Event &event, Status status) {
    ++total_rejected_;
    return Emit(context, kOutputRejected, event.seq, event.request,
                event.generation, event.row, 0, kNoToken, 0, event.target_steps,
                event.credit, event.cancel_seq, status, 0);
  }

  int32_t EnsureCache(const std::shared_ptr<MetaRunContext> &context) {
    constexpr int64_t kCacheElements =
        kLayers * kPhysicalBlocks * kBlockSize * kKvHeads * kHeadDim;
    if (state_key_ != nullptr || state_value_ != nullptr) {
      return IsTensor(state_key_, TensorDataType::DT_BF16, kCacheElements) &&
                     IsTensor(state_value_, TensorDataType::DT_BF16,
                              kCacheElements)
                 ? FLOW_FUNC_SUCCESS
                 : FLOW_FUNC_FAILED;
    }
    const std::vector<int64_t> shape = {
        static_cast<int64_t>(kLayers),
        static_cast<int64_t>(kPhysicalBlocks),
        static_cast<int64_t>(kBlockSize),
        static_cast<int64_t>(kKvHeads),
        static_cast<int64_t>(kHeadDim)};
    auto key_tensor =
        FlowBufferFactory::AllocTensor(shape, TensorDataType::DT_BF16);
    auto value_tensor =
        FlowBufferFactory::AllocTensor(shape, TensorDataType::DT_BF16);
    if (key_tensor == nullptr || value_tensor == nullptr) {
      return FLOW_FUNC_FAILED;
    }
    auto key_message = context->ToFlowMsg(key_tensor);
    auto value_message = context->ToFlowMsg(value_tensor);
    if (!IsTensor(key_message, TensorDataType::DT_BF16, kCacheElements) ||
        !IsTensor(value_message, TensorDataType::DT_BF16, kCacheElements)) {
      return FLOW_FUNC_FAILED;
    }
    std::memset(key_message->GetTensor()->GetData(), 0,
                key_message->GetTensor()->GetDataSize());
    std::memset(value_message->GetTensor()->GetData(), 0,
                value_message->GetTensor()->GetDataSize());
    state_key_ = key_message;
    state_value_ = value_message;
    return FLOW_FUNC_SUCCESS;
  }

  bool ClearCacheRow(int32_t row) {
    if (!ValidRow(row)) return false;
    constexpr size_t kBlockBytes = kBlockSize * kKvHeads * kHeadDim * 2;
    for (auto &cache : {state_key_, state_value_}) {
      if (!cache) return false;
      auto *data = static_cast<uint8_t *>(cache->GetTensor()->GetData());
      for (size_t layer = 0; layer < kLayers; ++layer) {
        for (size_t page = 0; page < kBlocksPerRequest; ++page) {
          const size_t physical = static_cast<size_t>(row) * kBlocksPerRequest + page;
          const size_t offset = (layer * kPhysicalBlocks + physical) * kBlockBytes;
          std::memset(data + offset, 0, kBlockBytes);
        }
      }
    }
    return true;
  }

  int32_t HandleAdmit(const std::shared_ptr<MetaRunContext> &context,
                      const Event &event) {
    const bool staged_cohort = event.reserved0 > 0;
    if (event.row != -1 || event.request <= 0 || event.generation <= 0 ||
        event.prompt_len <= 0 || event.prompt_len > kMaxPromptTokens ||
        event.target_steps <= 0 || event.target_steps > kMaxTargetSteps ||
        event.prompt_len + event.target_steps > kLogicalCapacity ||
        event.credit < 0 || event.credit > event.target_steps ||
        event.eos_token < 0 || event.eos_token >= 152064 ||
        (event.flags & ~kFlagIgnoreEos) != 0 || event.reserved2 != 0 ||
        event.reserved3 != 0 ||
        (!staged_cohort && event.reserved1 != 0) ||
        (staged_cohort &&
         (event.reserved1 <= 0 || event.reserved1 > static_cast<int64_t>(kRows)))) {
      return Reject(context, event, kStatusBadPrompt);
    }
    for (size_t index = 0; index < kMaxPromptTokens; ++index) {
      if (index < static_cast<size_t>(event.prompt_len)) {
        if (event.prompt[index] < 0 || event.prompt[index] >= 152064) {
          return Reject(context, event, kStatusBadPrompt);
        }
      } else if (event.prompt[index] != 0) {
        return Reject(context, event, kStatusBadPrompt);
      }
    }
    for (size_t index = 0; index < kRows; ++index) {
      if (rows_[index].active &&
          (rows_[index].request == event.request ||
           rows_[index].generation == event.generation)) {
        return Reject(context, event, kStatusBusy);
      }
    }
    int64_t staged_count = 0;
    int64_t available_count = 0;
    for (const auto &row : rows_) {
      if (!row.active) {
        ++available_count;
      } else if (row.staged) {
        if (!staged_cohort || row.cohort_id != event.reserved0 ||
            row.cohort_size != event.reserved1) {
          return Reject(context, event, kStatusBadCohort);
        }
        ++staged_count;
      }
    }
    if (staged_cohort &&
        (staged_count >= event.reserved1 ||
         staged_count + available_count < event.reserved1)) {
      return Reject(context, event, kStatusBadCohort);
    }
    int32_t assigned = -1;
    for (size_t index = 0; index < kRows; ++index) {
      if (!rows_[index].active && event.generation > rows_[index].last_generation) {
        assigned = static_cast<int32_t>(index);
        break;
      }
    }
    if (assigned < 0) return Reject(context, event, kStatusNoCapacity);
    if (EnsureCache(context) != FLOW_FUNC_SUCCESS ||
        !ClearCacheRow(assigned)) {
      return Reject(context, event, kStatusAllocationFailure);
    }
    auto &row = rows_[static_cast<size_t>(assigned)];
    const int64_t last_generation = row.last_generation;
    row = Row{};
    row.active = true;
    row.staged = staged_cohort;
    row.request = event.request;
    row.generation = event.generation;
    row.last_generation = std::max(last_generation, event.generation);
    row.prompt_len = event.prompt_len;
    row.target_steps = event.target_steps;
    row.credit = event.credit;
    row.eos_token = event.eos_token;
    row.flags = event.flags;
    row.cohort_id = event.reserved0;
    row.cohort_size = event.reserved1;
    row.token = 0;
    row.position = 0;
    row.prompt = event.prompt;
    ++total_admitted_;
    last_event_row_ = assigned;
    if (staged_cohort && staged_count + 1 == event.reserved1) {
      for (auto &candidate : rows_) {
        if (candidate.active && candidate.staged &&
            candidate.cohort_id == event.reserved0) {
          candidate.staged = false;
        }
      }
    }
    return EmitRow(context, kOutputAdmitAck, event.seq, assigned, kNoToken, 0,
                   kStatusOk);
  }

  int32_t HandleCredit(const std::shared_ptr<MetaRunContext> &context,
                       const Event &event) {
    if (!ValidRow(event.row)) return Reject(context, event, kStatusBadRow);
    auto &row = rows_[static_cast<size_t>(event.row)];
    if (!row.active) return Reject(context, event, kStatusNotActive);
    if (row.request != event.request || row.generation != event.generation) {
      return Reject(context, event, kStatusBadGeneration);
    }
    if (event.credit <= 0 || event.credit > row.target_steps - row.commits - row.credit) {
      return Reject(context, event, kStatusBadCredit);
    }
    row.credit += event.credit;
    last_event_row_ = event.row;
    return EmitRow(context, kOutputCreditAck, event.seq, event.row, kNoToken, 0,
                   kStatusOk);
  }

  int32_t HandleCancel(const std::shared_ptr<MetaRunContext> &context,
                       const Event &event) {
    if (!ValidRow(event.row)) return Reject(context, event, kStatusBadRow);
    auto &row = rows_[static_cast<size_t>(event.row)];
    if (!row.active) return Reject(context, event, kStatusNotActive);
    if (row.request != event.request || row.generation != event.generation) {
      return Reject(context, event, kStatusBadGeneration);
    }
    if (event.cancel_seq <= row.cancel_seq || event.cancel_seq <= 0) {
      return Reject(context, event, kStatusStaleCancel);
    }
    row.cancel_seq = event.cancel_seq;
    row.active = false;
    ++total_retired_;
    last_event_row_ = event.row;
    if (EmitRow(context, kOutputRetireCancelled, event.seq, event.row, kNoToken,
                event.cancel_seq, kStatusOk, 3) != FLOW_FUNC_SUCCESS) {
      return FLOW_FUNC_FAILED;
    }
    if (!HasActiveRow()) {
      ++quiescent_count_;
      if (EmitRow(context, kOutputQuiescent, event.seq, event.row, kNoToken, 0,
                  kStatusOk) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
      }
    }
    return FLOW_FUNC_SUCCESS;
  }

  int32_t HandleEvent(const std::shared_ptr<MetaRunContext> &context,
                      const Event &event, bool &shutdown) {
    shutdown = false;
    if (event.owner != owner_) return Reject(context, event, kStatusWrongOwner);
    if (has_last_event_ && event.seq == last_event_seq_ &&
        SameEvent(event, last_event_)) {
      return EmitDuplicate(context, event);
    }
    if (event.seq <= last_event_seq_) {
      return Reject(context, event, kStatusStaleEvent);
    }
    last_event_ = event;
    last_event_seq_ = event.seq;
    has_last_event_ = true;
    ++total_events_;
    if (event.type == kEventAdmit) return HandleAdmit(context, event);
    if (event.type == kEventCredit) return HandleCredit(context, event);
    if (event.type == kEventCancel) return HandleCancel(context, event);
    if (event.type == kEventShutdown) {
      if (HasActiveRow()) return Reject(context, event, kStatusBusy);
      ++quiescent_count_;
      shutdown = true;
      return Emit(context, kOutputShutdown, event.seq, 0, 0, -1, 0, kNoToken, 0,
                  0, 0, 0, kStatusOk, 0, true);
    }
    return Reject(context, event, kStatusUnknownEvent);
  }

  int32_t RunQuantum(const std::shared_ptr<MetaRunContext> &context) {
    auto token = context->AllocTensorMsg({static_cast<int64_t>(kRows), 1},
                                         TensorDataType::DT_INT64);
    auto position = context->AllocTensorMsg({static_cast<int64_t>(kRows)},
                                             TensorDataType::DT_INT64);
    auto length = context->AllocTensorMsg({static_cast<int64_t>(kRows), 1},
                                          TensorDataType::DT_INT32);
    auto slot = context->AllocTensorMsg({static_cast<int64_t>(kRows)},
                                         TensorDataType::DT_INT32);
    auto active = context->AllocTensorMsg({static_cast<int64_t>(kRows)},
                                           TensorDataType::DT_INT32);
    auto block_table = context->AllocTensorMsg(
        {static_cast<int64_t>(kRows), static_cast<int64_t>(kBlocksPerRequest)},
        TensorDataType::DT_INT32);
    if (!IsTensor(token, TensorDataType::DT_INT64, kRows) ||
        !IsTensor(position, TensorDataType::DT_INT64, kRows) ||
        !IsTensor(length, TensorDataType::DT_INT32, kRows) ||
        !IsTensor(slot, TensorDataType::DT_INT32, kRows) ||
        !IsTensor(active, TensorDataType::DT_INT32, kRows) ||
        !IsTensor(block_table, TensorDataType::DT_INT32,
                  kRows * kBlocksPerRequest) ||
        !IsTensor(state_key_, TensorDataType::DT_BF16,
                  kLayers * kPhysicalBlocks * kBlockSize * kKvHeads * kHeadDim) ||
        !IsTensor(state_value_, TensorDataType::DT_BF16,
                  kLayers * kPhysicalBlocks * kBlockSize * kKvHeads * kHeadDim)) {
      return FLOW_FUNC_FAILED;
    }
    auto *tokens = static_cast<int64_t *>(token->GetTensor()->GetData());
    auto *positions = static_cast<int64_t *>(position->GetTensor()->GetData());
    auto *lengths = static_cast<int32_t *>(length->GetTensor()->GetData());
    auto *slots = static_cast<int32_t *>(slot->GetTensor()->GetData());
    auto *active_values = static_cast<int32_t *>(active->GetTensor()->GetData());
    auto *blocks = static_cast<int32_t *>(block_table->GetTensor()->GetData());
    for (size_t index = 0; index < kRows; ++index) {
      auto &row = rows_[index];
      const int64_t pos = row.prompt_index < row.prompt_len
                              ? row.prompt_index
                              : row.position;
      const bool eligible = Eligible(index);
      tokens[index] = row.prompt_index < row.prompt_len
                          ? row.prompt[static_cast<size_t>(row.prompt_index)]
                          : row.token;
      positions[index] = pos;
      lengths[index] = static_cast<int32_t>(pos + 1);
      slots[index] = ComputeSlot(static_cast<int32_t>(index), pos);
      active_values[index] = eligible ? 1 : 0;
      for (size_t page = 0; page < kBlocksPerRequest; ++page) {
        blocks[index * kBlocksPerRequest + page] =
            static_cast<int32_t>(index * kBlocksPerRequest + page);
      }
      if (tokens[index] < 0 || tokens[index] >= 152064 || slots[index] < 0) {
        return FLOW_FUNC_FAILED;
      }
    }
    std::vector<std::shared_ptr<FlowMsg>> outputs;
    // This order follows the audited AIR Data-node order, not the Python
    // function signature. Host never constructs this per-token input list.
    const auto ret = context->RunFlowModel(
        "decode_graph_0",
        {token, position, length, state_key_, slot, active, block_table,
         state_value_},
        outputs, kRunModelTimeoutMs);
    ++aicore_calls_;
    if (ret != FLOW_FUNC_SUCCESS || outputs.size() != 4 ||
        !IsTensor(outputs[0], TensorDataType::DT_INT64, kRows) ||
        !IsTensor(outputs[1], TensorDataType::DT_BF16,
                  kLayers * kPhysicalBlocks * kBlockSize * kKvHeads * kHeadDim) ||
        !IsTensor(outputs[2], TensorDataType::DT_BF16,
                  kLayers * kPhysicalBlocks * kBlockSize * kKvHeads * kHeadDim) ||
        !IsTensor(outputs[3], TensorDataType::DT_INT64, kRows)) {
      return FLOW_FUNC_FAILED;
    }
    const auto *generated =
        static_cast<const int64_t *>(outputs[0]->GetTensor()->GetData());
    const auto *next_positions =
        static_cast<const int64_t *>(outputs[3]->GetTensor()->GetData());
    state_key_ = outputs[1];
    state_value_ = outputs[2];
    int64_t committed = 0;
    for (size_t index = 0; index < kRows; ++index) {
      auto &row = rows_[index];
      if (!row.active) continue;
      if (!active_values[index]) {
        if (next_positions[index] != row.position) return FLOW_FUNC_FAILED;
        continue;
      }
      if (next_positions[index] != row.position + 1) return FLOW_FUNC_FAILED;
      row.position = next_positions[index];
      row.token = generated[index];
      const bool prompt_step = row.prompt_index < row.prompt_len;
      if (prompt_step) ++row.prompt_index;
      const bool commit = !prompt_step || row.prompt_index == row.prompt_len;
      if (!commit) continue;
      if (generated[index] < 0 || generated[index] >= 152064) return FLOW_FUNC_FAILED;
      ++row.commits;
      --row.credit;
      ++total_commits_;
      ++next_commit_seq_;
      ++committed;
      if (EmitRow(context, kOutputCommit, last_event_seq_,
                  static_cast<int32_t>(index), generated[index], 0, kStatusOk) !=
          FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
      }
      const bool eos = generated[index] == row.eos_token &&
                       (row.flags & kFlagIgnoreEos) == 0;
      if (eos || row.commits == row.target_steps) {
        row.active = false;
        ++total_retired_;
        if (EmitRow(context, kOutputRetireComplete, last_event_seq_,
                    static_cast<int32_t>(index), generated[index], 0,
                    kStatusOk, eos ? 1 : 2) != FLOW_FUNC_SUCCESS) {
          return FLOW_FUNC_FAILED;
        }
      }
    }
    if (committed > 0 && !HasActiveRow()) {
      ++quiescent_count_;
      if (EmitRow(context, kOutputQuiescent, last_event_seq_, last_event_row_,
                  kNoToken, 0, kStatusOk) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
      }
    }
    return FLOW_FUNC_SUCCESS;
  }

  int64_t owner_ = 0;
  int64_t last_event_seq_ = 0;
  Event last_event_{};
  bool has_last_event_ = false;
  int64_t last_event_row_ = -1;
  int64_t total_events_ = 0;
  int64_t total_duplicates_ = 0;
  int64_t total_rejected_ = 0;
  int64_t total_admitted_ = 0;
  int64_t total_retired_ = 0;
  int64_t total_commits_ = 0;
  int64_t aicore_calls_ = 0;
  int64_t next_commit_seq_ = 0;
  int64_t quiescent_count_ = 0;
  std::array<Row, kRows> rows_{};
  std::shared_ptr<FlowMsg> state_key_;
  std::shared_ptr<FlowMsg> state_value_;
};

FLOW_FUNC_REGISTRAR(PersistentDecoderP3)
    .RegProcFunc("persistent_decoder_p3", &PersistentDecoderP3::Proc);
}  // namespace FlowFunc
