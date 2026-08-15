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
constexpr size_t kEventElements = 16;
constexpr size_t kOutputElements = 12;
constexpr int64_t kTargetSteps = 256;
constexpr useconds_t kSyntheticCommitPaceUs = 1000;
constexpr int32_t kRunModelTimeoutMs = 300000;
constexpr int64_t kEventAdmit = 1;
constexpr int64_t kEventCredit = 2;
constexpr int64_t kEventShutdown = 3;
constexpr int64_t kOutputAdmitAck = 1;
constexpr int64_t kOutputCommit = 2;
constexpr int64_t kOutputRetireComplete = 3;
constexpr int64_t kOutputQuiescent = 4;
constexpr int64_t kOutputCreditAck = 5;
constexpr int64_t kOutputShutdown = 6;
constexpr int64_t kOutputRejected = 7;

enum Status : int64_t {
  kStatusOk = 0,
  kStatusWrongOwner = 1,
  kStatusStaleEvent = 2,
  kStatusBusy = 3,
  kStatusBadCohort = 4,
  kStatusBadBatch = 5,
  kStatusBadCredit = 6,
  kStatusUnknownEvent = 7,
  kStatusBadMessage = 8,
  kStatusModelFailure = 9,
  kStatusModelOutput = 10,
};

struct Event {
  int64_t owner = 0;
  int64_t type = 0;
  int64_t seq = 0;
  int64_t cohort = 0;
  int64_t batch = 0;
  std::array<int64_t, kRows> credits{};
  std::array<int64_t, kRows> seeds{};
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
  event.cohort = values[3];
  event.batch = values[4];
  for (size_t row = 0; row < kRows; ++row) {
    event.credits[row] = values[5 + row];
    event.seeds[row] = values[9 + row];
  }
  return true;
}
}  // namespace

class PersistentRecurrenceP1 : public MetaMultiFunc {
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
          if (Emit(context, kOutputRejected, last_event_seq_, cohort_, -1, 0,
                   0, 0, 0, 0, kStatusBadMessage) != FLOW_FUNC_SUCCESS) {
            return FLOW_FUNC_FAILED;
          }
        } else {
          bool shutdown = false;
          const int32_t handled = HandleEvent(context, event, shutdown);
          if (handled != FLOW_FUNC_SUCCESS) {
            return handled;
          }
          if (shutdown) {
            return FLOW_FUNC_SUCCESS;
          }
        }
      } else if (dequeue != FLOW_FUNC_ERR_TIME_OUT_ERROR) {
        return dequeue;
      }
      if (HasEligibleRow() && RunQuantum(context) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
      }
    }
  }

 private:
  void ResetState() {
    last_event_seq_ = 0;
    cohort_event_seq_ = 0;
    cohort_ = 0;
    last_cohort_ = 0;
    batch_ = 0;
    total_commits_ = 0;
    aicore_calls_ = 0;
    quiescent_count_ = 0;
    active_.fill(false);
    credits_.fill(0);
    commits_.fill(0);
    seeds_.fill(0);
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

  int64_t TotalCredits() const {
    int64_t count = 0;
    for (int64_t credit : credits_) count += credit;
    return count;
  }

  int32_t Emit(const std::shared_ptr<MetaRunContext> &context,
               int64_t output_type, int64_t event_seq, int64_t cohort,
               int64_t row, int64_t commit_seq, int64_t state, int64_t token,
               int64_t remaining, int64_t credit_remaining, int64_t status,
               bool eos = false) {
    auto output = context->AllocTensorMsg(
        {static_cast<int64_t>(kOutputElements)}, TensorDataType::DT_INT64);
    if (!IsTensor(output, TensorDataType::DT_INT64, kOutputElements)) {
      return FLOW_FUNC_FAILED;
    }
    const int64_t values[kOutputElements] = {
        owner_, output_type, event_seq, cohort, row, commit_seq,
        state, token, remaining, credit_remaining, status, 0};
    std::memcpy(output->GetTensor()->GetData(), values, sizeof(values));
    output->SetTransactionId(static_cast<uint64_t>(event_seq));
    if (eos) {
      output->SetFlowFlags(static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS));
    }
    return context->SetOutput(0, output);
  }

  int32_t Reject(const std::shared_ptr<MetaRunContext> &context,
                 const Event &event, Status status) {
    return Emit(context, kOutputRejected, event.seq, event.cohort, -1, 0, 0,
                0, ActiveRows(), TotalCredits(), status);
  }

  int32_t EmitQuiescent(const std::shared_ptr<MetaRunContext> &context) {
    ++quiescent_count_;
    return Emit(context, kOutputQuiescent, cohort_event_seq_, cohort_, -1, 0,
                0, 0, 0, 0, quiescent_count_);
  }

  int32_t HandleEvent(const std::shared_ptr<MetaRunContext> &context,
                      const Event &event, bool &shutdown) {
    shutdown = false;
    if (event.owner != owner_) return Reject(context, event, kStatusWrongOwner);
    if (event.seq <= last_event_seq_) {
      return Reject(context, event, kStatusStaleEvent);
    }
    last_event_seq_ = event.seq;
    if (event.type == kEventAdmit) {
      if (HasActiveRow()) return Reject(context, event, kStatusBusy);
      if (event.cohort <= last_cohort_) {
        return Reject(context, event, kStatusBadCohort);
      }
      if (event.batch != 1 && event.batch != 4) {
        return Reject(context, event, kStatusBadBatch);
      }
      auto state = context->AllocTensorMsg(
          {static_cast<int64_t>(kRows)}, TensorDataType::DT_INT32);
      if (!IsTensor(state, TensorDataType::DT_INT32, kRows)) {
        return FLOW_FUNC_FAILED;
      }
      auto *values = static_cast<int32_t *>(state->GetTensor()->GetData());
      for (size_t row = 0; row < kRows; ++row) {
        const bool admitted = row < static_cast<size_t>(event.batch);
        if ((admitted && (event.credits[row] <= 0 ||
                          event.credits[row] > kTargetSteps)) ||
            (!admitted && event.credits[row] != 0) ||
            event.seeds[row] < std::numeric_limits<int32_t>::min() ||
            event.seeds[row] > std::numeric_limits<int32_t>::max() -
                                   kTargetSteps) {
          return Reject(context, event, kStatusBadCredit);
        }
        active_[row] = admitted;
        credits_[row] = event.credits[row];
        commits_[row] = 0;
        seeds_[row] = event.seeds[row];
        values[row] = admitted ? static_cast<int32_t>(event.seeds[row]) : 0;
      }
      state_ = state;
      cohort_ = event.cohort;
      last_cohort_ = event.cohort;
      cohort_event_seq_ = event.seq;
      batch_ = event.batch;
      return Emit(context, kOutputAdmitAck, event.seq, cohort_, -1, 0, 0, 0,
                  kTargetSteps * batch_, TotalCredits(), kStatusOk);
    }
    if (event.type == kEventCredit) {
      if (!HasActiveRow() || event.cohort != cohort_) {
        return Reject(context, event, kStatusBadCohort);
      }
      for (size_t row = 0; row < kRows; ++row) {
        if (event.credits[row] < 0 ||
            (!active_[row] && event.credits[row] != 0) ||
            credits_[row] + event.credits[row] >
                kTargetSteps - commits_[row]) {
          return Reject(context, event, kStatusBadCredit);
        }
      }
      for (size_t row = 0; row < kRows; ++row) {
        credits_[row] += event.credits[row];
      }
      return Emit(context, kOutputCreditAck, event.seq, cohort_, -1, 0, 0, 0,
                  ActiveRows(), TotalCredits(), kStatusOk);
    }
    if (event.type == kEventShutdown) {
      if (HasActiveRow()) return Reject(context, event, kStatusBusy);
      shutdown = true;
      return Emit(context, kOutputShutdown, event.seq, cohort_, -1,
                  aicore_calls_, total_commits_, total_commits_, 0, 0,
                  kStatusOk, true);
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
    const int32_t run = context->RunFlowModel(
        "recurrence_graph", {state_, delta}, outputs, kRunModelTimeoutMs);
    if (run != FLOW_FUNC_SUCCESS) {
      return FLOW_FUNC_FAILED;
    }
    if (outputs.size() != 1 ||
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
    for (size_t row = 0; row < kRows; ++row) {
      if (!eligible[row]) continue;
      ++eligible_rows;
      ++commits_[row];
      --credits_[row];
      ++total_commits_;
      const int64_t remaining = kTargetSteps - commits_[row];
      if (Emit(context, kOutputCommit, cohort_event_seq_, cohort_, row,
               commits_[row], next_values[row], next_values[row], remaining,
               credits_[row], kStatusOk) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
      }
      if (remaining == 0) {
        active_[row] = false;
        if (Emit(context, kOutputRetireComplete, cohort_event_seq_, cohort_,
                 row, commits_[row], next_values[row], next_values[row], 0,
                 credits_[row], kStatusOk) != FLOW_FUNC_SUCCESS) {
          return FLOW_FUNC_FAILED;
        }
      }
    }
    if (!HasActiveRow() && EmitQuiescent(context) != FLOW_FUNC_SUCCESS) {
      return FLOW_FUNC_FAILED;
    }
    usleep(static_cast<useconds_t>(eligible_rows) * kSyntheticCommitPaceUs);
    return FLOW_FUNC_SUCCESS;
  }

  int64_t owner_ = 0;
  int64_t last_event_seq_ = 0;
  int64_t cohort_event_seq_ = 0;
  int64_t cohort_ = 0;
  int64_t last_cohort_ = 0;
  int64_t batch_ = 0;
  int64_t total_commits_ = 0;
  int64_t aicore_calls_ = 0;
  int64_t quiescent_count_ = 0;
  std::array<bool, kRows> active_{};
  std::array<int64_t, kRows> credits_{};
  std::array<int64_t, kRows> commits_{};
  std::array<int64_t, kRows> seeds_{};
  std::shared_ptr<FlowMsg> state_;
};

FLOW_FUNC_REGISTRAR(PersistentRecurrenceP1)
    .RegProcFunc("persistent_recurrence_p1", &PersistentRecurrenceP1::Proc);
}  // namespace FlowFunc
