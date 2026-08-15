#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <unistd.h>

#include "flow_func/meta_multi_func.h"

namespace FlowFunc {
namespace {
constexpr size_t kRows = 4;
constexpr size_t kEventElements = 12;
constexpr size_t kOutputElements = 16;
constexpr int64_t kMaxTargetSteps = 256;
constexpr useconds_t kSyntheticCommitPaceUs = 1000;
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

enum Status : int64_t {
  kStatusOk = 0,
  kStatusWrongOwner = 1,
  kStatusStaleEvent = 2,
  kStatusNoCapacity = 3,
  kStatusBadGeneration = 4,
  kStatusBadRow = 5,
  kStatusBadTarget = 6,
  kStatusBadCredit = 7,
  kStatusStaleCancel = 8,
  kStatusNotActive = 9,
  kStatusUnknownEvent = 10,
  kStatusBadMessage = 11,
  kStatusDuplicate = 12,
  kStatusBusy = 13,
};

struct Event {
  int64_t owner = 0;
  int64_t type = 0;
  int64_t seq = 0;
  int64_t request = 0;
  int64_t generation = 0;
  int64_t row = -1;
  int64_t target_steps = 0;
  int64_t credit = 0;
  int64_t cancel_seq = 0;
  int64_t seed = 0;
  int64_t flags = 0;
  int64_t reserved = 0;
};

bool IsTensor(const std::shared_ptr<FlowMsg> &message, TensorDataType data_type,
              int64_t elements) {
  if (message == nullptr || message->GetRetCode() != FLOW_FUNC_SUCCESS ||
      message->GetMsgType() != MsgType::MSG_TYPE_TENSOR_DATA) {
    return false;
  }
  const auto *tensor = message->GetTensor();
  return tensor != nullptr && tensor->GetDataType() == data_type &&
         tensor->GetElementCnt() == elements && tensor->GetData() != nullptr;
}

bool ReadEvent(const std::shared_ptr<FlowMsg> &message, Event &event) {
  if (!IsTensor(message, TensorDataType::DT_INT64, kEventElements) ||
      message->GetTensor()->GetDataSize() !=
          kEventElements * sizeof(int64_t)) {
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
  event.target_steps = values[6];
  event.credit = values[7];
  event.cancel_seq = values[8];
  event.seed = values[9];
  event.flags = values[10];
  event.reserved = values[11];
  return true;
}

bool SameEvent(const Event &left, const Event &right) {
  return left.owner == right.owner && left.type == right.type &&
         left.seq == right.seq && left.request == right.request &&
         left.generation == right.generation && left.row == right.row &&
         left.target_steps == right.target_steps &&
         left.credit == right.credit &&
         left.cancel_seq == right.cancel_seq && left.seed == right.seed &&
         left.flags == right.flags && left.reserved == right.reserved;
}
}  // namespace

class PersistentCancelP2 : public MetaMultiFunc {
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
                   0, 0, 0, 0, kStatusBadMessage) != FLOW_FUNC_SUCCESS) {
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
    last_event_seq_ = 0;
    has_last_event_ = false;
    last_event_row_ = -1;
    total_events_ = 0;
    total_duplicates_ = 0;
    total_rejected_ = 0;
    total_admitted_ = 0;
    total_retired_ = 0;
    total_cancelled_ = 0;
    total_commits_ = 0;
    aicore_calls_ = 0;
    quiescent_count_ = 0;
    active_.fill(false);
    requests_.fill(0);
    generations_.fill(0);
    last_generations_.fill(0);
    targets_.fill(0);
    credits_.fill(0);
    commits_.fill(0);
    seeds_.fill(0);
    last_cancel_seqs_.fill(0);
    state_.reset();
  }

  bool HasActiveRow() const {
    for (bool active : active_) {
      if (active) return true;
    }
    return false;
  }

  bool HasEligibleRow() const {
    for (size_t row = 0; row < kRows; ++row) {
      if (active_[row] && credits_[row] > 0) return true;
    }
    return false;
  }

  int64_t ActiveRows() const {
    int64_t count = 0;
    for (bool active : active_) count += active ? 1 : 0;
    return count;
  }

  bool ValidRow(int64_t row) const {
    return row >= 0 && row < static_cast<int64_t>(kRows);
  }

  int32_t Emit(const std::shared_ptr<MetaRunContext> &context,
               int64_t output_type, int64_t event_seq, int64_t request,
               int64_t generation, int64_t row, int64_t commit_seq,
               int64_t state, int64_t remaining, int64_t credit,
               int64_t cancel_seq, int64_t status, bool eos = false) {
    auto output = context->AllocTensorMsg(
        {static_cast<int64_t>(kOutputElements)}, TensorDataType::DT_INT64);
    if (!IsTensor(output, TensorDataType::DT_INT64, kOutputElements)) {
      return FLOW_FUNC_FAILED;
    }
    const int64_t values[kOutputElements] = {
        owner_, output_type, event_seq, request, generation, row,
        commit_seq, state, remaining, credit, cancel_seq, status,
        aicore_calls_, total_commits_, total_retired_, quiescent_count_};
    std::memcpy(output->GetTensor()->GetData(), values, sizeof(values));
    output->SetTransactionId(static_cast<uint64_t>(event_seq));
    if (eos) {
      output->SetFlowFlags(static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS));
    }
    return context->SetOutput(0, output);
  }

  int32_t EmitRow(const std::shared_ptr<MetaRunContext> &context,
                  int64_t output_type, int64_t event_seq, int64_t row,
                  int64_t cancel_seq, int64_t status) {
    if (!ValidRow(row)) {
      return Emit(context, output_type, event_seq, 0, 0, row, 0, 0, 0, 0,
                  cancel_seq, status);
    }
    const size_t index = static_cast<size_t>(row);
    const int64_t state =
        IsTensor(state_, TensorDataType::DT_INT32, kRows)
            ? static_cast<const int32_t *>(state_->GetTensor()->GetData())[index]
            : 0;
    return Emit(context, output_type, event_seq, requests_[index],
                generations_[index], row, commits_[index], state,
                targets_[index] - commits_[index], credits_[index], cancel_seq,
                status);
  }

  int32_t EmitQuiescent(const std::shared_ptr<MetaRunContext> &context,
                        int64_t event_seq, int64_t row) {
    ++quiescent_count_;
    return EmitRow(context, kOutputQuiescent, event_seq, row, 0, kStatusOk);
  }

  int32_t Reject(const std::shared_ptr<MetaRunContext> &context,
                 const Event &event, Status status) {
    ++total_rejected_;
    if (ValidRow(event.row)) {
      return EmitRow(context, kOutputRejected, event.seq, event.row,
                     event.cancel_seq, status);
    }
    return Emit(context, kOutputRejected, event.seq, event.request,
                event.generation, event.row, 0, event.seed,
                event.target_steps, event.credit, event.cancel_seq, status);
  }

  int32_t EmitDuplicate(const std::shared_ptr<MetaRunContext> &context,
                        const Event &event) {
    ++total_duplicates_;
    if (ValidRow(last_event_row_)) {
      return EmitRow(context, kOutputCumulativeAck, event.seq, last_event_row_,
                     event.cancel_seq, kStatusDuplicate);
    }
    return Emit(context, kOutputCumulativeAck, event.seq, event.request,
                event.generation, event.row, 0, event.seed,
                event.target_steps, event.credit, event.cancel_seq,
                kStatusDuplicate);
  }

  int32_t AllocateState(const std::shared_ptr<MetaRunContext> &context,
                        size_t assigned_row, int64_t seed) {
    auto state = context->AllocTensorMsg(
        {static_cast<int64_t>(kRows)}, TensorDataType::DT_INT32);
    if (!IsTensor(state, TensorDataType::DT_INT32, kRows)) {
      return FLOW_FUNC_FAILED;
    }
    auto *destination = static_cast<int32_t *>(state->GetTensor()->GetData());
    const int32_t *source =
        IsTensor(state_, TensorDataType::DT_INT32, kRows)
            ? static_cast<const int32_t *>(state_->GetTensor()->GetData())
            : nullptr;
    for (size_t row = 0; row < kRows; ++row) {
      destination[row] = source == nullptr ? 0 : source[row];
    }
    destination[assigned_row] = static_cast<int32_t>(seed);
    state_ = state;
    return FLOW_FUNC_SUCCESS;
  }

  int32_t HandleAdmit(const std::shared_ptr<MetaRunContext> &context,
                      const Event &event) {
    if (event.request <= 0 || event.generation <= 0 || event.row != -1 ||
        event.target_steps <= 0 || event.target_steps > kMaxTargetSteps ||
        event.credit < 0 || event.credit > event.target_steps ||
        event.seed < std::numeric_limits<int32_t>::min() ||
        event.seed >
            std::numeric_limits<int32_t>::max() - event.target_steps) {
      return Reject(context, event, kStatusBadTarget);
    }
    for (size_t row = 0; row < kRows; ++row) {
      if (active_[row] &&
          (requests_[row] == event.request ||
           generations_[row] == event.generation)) {
        return Reject(context, event, kStatusBusy);
      }
    }
    int64_t assigned = -1;
    bool free_row = false;
    for (size_t row = 0; row < kRows; ++row) {
      if (active_[row]) continue;
      free_row = true;
      if (event.generation > last_generations_[row]) {
        assigned = static_cast<int64_t>(row);
        break;
      }
    }
    if (assigned < 0) {
      return Reject(context, event,
                    free_row ? kStatusBadGeneration : kStatusNoCapacity);
    }
    const size_t row = static_cast<size_t>(assigned);
    if (AllocateState(context, row, event.seed) != FLOW_FUNC_SUCCESS) {
      return FLOW_FUNC_FAILED;
    }
    active_[row] = true;
    requests_[row] = event.request;
    generations_[row] = event.generation;
    last_generations_[row] = event.generation;
    targets_[row] = event.target_steps;
    credits_[row] = event.credit;
    commits_[row] = 0;
    seeds_[row] = event.seed;
    last_cancel_seqs_[row] = 0;
    ++total_admitted_;
    last_event_row_ = assigned;
    return EmitRow(context, kOutputAdmitAck, event.seq, assigned, 0,
                   kStatusOk);
  }

  int32_t HandleCredit(const std::shared_ptr<MetaRunContext> &context,
                       const Event &event) {
    if (!ValidRow(event.row)) return Reject(context, event, kStatusBadRow);
    const size_t row = static_cast<size_t>(event.row);
    if (!active_[row]) return Reject(context, event, kStatusNotActive);
    if (requests_[row] != event.request ||
        generations_[row] != event.generation) {
      return Reject(context, event, kStatusBadGeneration);
    }
    if (event.credit <= 0 ||
        credits_[row] + event.credit > targets_[row] - commits_[row]) {
      return Reject(context, event, kStatusBadCredit);
    }
    credits_[row] += event.credit;
    last_event_row_ = event.row;
    return EmitRow(context, kOutputCreditAck, event.seq, event.row, 0,
                   kStatusOk);
  }

  int32_t HandleCancel(const std::shared_ptr<MetaRunContext> &context,
                       const Event &event) {
    if (!ValidRow(event.row)) return Reject(context, event, kStatusBadRow);
    const size_t row = static_cast<size_t>(event.row);
    if (!active_[row]) {
      return Reject(context, event,
                    requests_[row] == event.request &&
                            generations_[row] == event.generation
                        ? kStatusNotActive
                        : kStatusBadGeneration);
    }
    if (requests_[row] != event.request ||
        generations_[row] != event.generation) {
      return Reject(context, event, kStatusBadGeneration);
    }
    if (event.cancel_seq <= 0 ||
        event.cancel_seq <= last_cancel_seqs_[row]) {
      return Reject(context, event, kStatusStaleCancel);
    }
    last_cancel_seqs_[row] = event.cancel_seq;
    active_[row] = false;
    ++total_retired_;
    ++total_cancelled_;
    last_event_row_ = event.row;
    if (EmitRow(context, kOutputRetireCancelled, event.seq, event.row,
                event.cancel_seq, kStatusOk) != FLOW_FUNC_SUCCESS) {
      return FLOW_FUNC_FAILED;
    }
    if (!HasActiveRow() &&
        EmitQuiescent(context, event.seq, event.row) != FLOW_FUNC_SUCCESS) {
      return FLOW_FUNC_FAILED;
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
    has_last_event_ = true;
    last_event_seq_ = event.seq;
    last_event_row_ = event.row;
    ++total_events_;

    if (event.type == kEventAdmit) return HandleAdmit(context, event);
    if (event.type == kEventCredit) return HandleCredit(context, event);
    if (event.type == kEventCancel) return HandleCancel(context, event);
    if (event.type == kEventShutdown) {
      if (HasActiveRow()) return Reject(context, event, kStatusBusy);
      shutdown = true;
      return Emit(context, kOutputShutdown, event.seq, 0, 0, -1,
                  aicore_calls_, total_commits_, total_events_,
                  total_duplicates_, total_cancelled_, kStatusOk, true);
    }
    return Reject(context, event, kStatusUnknownEvent);
  }

  int32_t RunQuantum(const std::shared_ptr<MetaRunContext> &context) {
    if (!IsTensor(state_, TensorDataType::DT_INT32, kRows)) {
      return FLOW_FUNC_FAILED;
    }
    auto delta = context->AllocTensorMsg(
        {static_cast<int64_t>(kRows)}, TensorDataType::DT_INT32);
    if (!IsTensor(delta, TensorDataType::DT_INT32, kRows)) {
      return FLOW_FUNC_FAILED;
    }
    std::array<bool, kRows> eligible{};
    auto *delta_values =
        static_cast<int32_t *>(delta->GetTensor()->GetData());
    const auto *state_values =
        static_cast<const int32_t *>(state_->GetTensor()->GetData());
    std::array<int32_t, kRows> old_values{};
    for (size_t row = 0; row < kRows; ++row) {
      eligible[row] = active_[row] && credits_[row] > 0;
      delta_values[row] = eligible[row] ? 1 : 0;
      old_values[row] = state_values[row];
    }
    std::vector<std::shared_ptr<FlowMsg>> outputs;
    if (context->RunFlowModel("recurrence_graph", {state_, delta}, outputs,
                              kRunModelTimeoutMs) != FLOW_FUNC_SUCCESS ||
        outputs.size() != 1 ||
        !IsTensor(outputs[0], TensorDataType::DT_INT32, kRows)) {
      return FLOW_FUNC_FAILED;
    }
    const auto *next_values =
        static_cast<const int32_t *>(outputs[0]->GetTensor()->GetData());
    for (size_t row = 0; row < kRows; ++row) {
      const int32_t expected = old_values[row] + (eligible[row] ? 1 : 0);
      if (next_values[row] != expected) return FLOW_FUNC_FAILED;
    }
    state_ = outputs[0];
    ++aicore_calls_;
    int64_t eligible_rows = 0;
    int64_t quiescent_row = -1;
    for (size_t row = 0; row < kRows; ++row) {
      if (!eligible[row]) continue;
      ++eligible_rows;
      ++commits_[row];
      --credits_[row];
      ++total_commits_;
      if (EmitRow(context, kOutputCommit, last_event_seq_, row, 0,
                  kStatusOk) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
      }
      if (commits_[row] == targets_[row]) {
        active_[row] = false;
        ++total_retired_;
        quiescent_row = static_cast<int64_t>(row);
        if (EmitRow(context, kOutputRetireComplete, last_event_seq_, row, 0,
                    kStatusOk) != FLOW_FUNC_SUCCESS) {
          return FLOW_FUNC_FAILED;
        }
      }
    }
    if (!HasActiveRow() && quiescent_row >= 0 &&
        EmitQuiescent(context, last_event_seq_, quiescent_row) !=
            FLOW_FUNC_SUCCESS) {
      return FLOW_FUNC_FAILED;
    }
    usleep(static_cast<useconds_t>(eligible_rows) * kSyntheticCommitPaceUs);
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
  int64_t total_cancelled_ = 0;
  int64_t total_commits_ = 0;
  int64_t aicore_calls_ = 0;
  int64_t quiescent_count_ = 0;
  std::array<bool, kRows> active_{};
  std::array<int64_t, kRows> requests_{};
  std::array<int64_t, kRows> generations_{};
  std::array<int64_t, kRows> last_generations_{};
  std::array<int64_t, kRows> targets_{};
  std::array<int64_t, kRows> credits_{};
  std::array<int64_t, kRows> commits_{};
  std::array<int64_t, kRows> seeds_{};
  std::array<int64_t, kRows> last_cancel_seqs_{};
  std::shared_ptr<FlowMsg> state_;
};

FLOW_FUNC_REGISTRAR(PersistentCancelP2)
    .RegProcFunc("persistent_cancel_p2", &PersistentCancelP2::Proc);
}  // namespace FlowFunc
