#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <unistd.h>

#include "flow_func/flow_func_log.h"
#include "flow_func/meta_multi_func.h"

namespace FlowFunc {
namespace {
constexpr size_t kEventElements = 8;
constexpr size_t kOutputElements = 8;
constexpr int64_t kEventAdmit = 1;
constexpr int64_t kEventCancel = 2;
constexpr int64_t kEventShutdown = 3;
constexpr int64_t kOutputAdmitAck = 1;
constexpr int64_t kOutputCommit = 2;
constexpr int64_t kOutputRetireComplete = 3;
constexpr int64_t kOutputRetireCancelled = 4;
constexpr int64_t kOutputQuiescent = 5;
constexpr int64_t kOutputShutdown = 6;
constexpr int64_t kOutputRejected = 7;
constexpr int64_t kMaxSyntheticQuanta = 1LL << 20;

enum Status : int64_t {
  kStatusOk = 0,
  kStatusWrongOwner = 1,
  kStatusStaleEvent = 2,
  kStatusBusy = 3,
  kStatusBadGeneration = 4,
  kStatusBadTarget = 5,
  kStatusStaleCancel = 6,
  kStatusUnknownEvent = 7,
  kStatusBadMessage = 8,
};

struct Event {
  int64_t owner = 0;
  int64_t type = 0;
  int64_t seq = 0;
  int64_t generation = 0;
  int64_t target_quanta = 0;
  int64_t cancel_seq = 0;
  int64_t seed = 0;
};

bool ReadEvent(const std::shared_ptr<FlowMsg> &message, Event &event) {
  if (message == nullptr || message->GetMsgType() != MsgType::MSG_TYPE_TENSOR_DATA) {
    return false;
  }
  auto *tensor = message->GetTensor();
  if (tensor == nullptr || tensor->GetDataType() != TensorDataType::DT_INT64 ||
      tensor->GetElementCnt() != static_cast<int64_t>(kEventElements) ||
      tensor->GetDataSize() != kEventElements * sizeof(int64_t) ||
      tensor->GetData() == nullptr) {
    return false;
  }
  const auto *values = static_cast<const int64_t *>(tensor->GetData());
  event.owner = values[0];
  event.type = values[1];
  event.seq = values[2];
  event.generation = values[3];
  event.target_quanta = values[4];
  event.cancel_seq = values[5];
  event.seed = values[6];
  return true;
}
}  // namespace

class PersistentControlP0 : public MetaMultiFunc {
 public:
  int32_t Init(const std::shared_ptr<MetaParams> &params) override {
    if (params == nullptr || params->GetInputNum() != 1 ||
        params->GetOutputNum() != 1 ||
        params->GetAttr("owner_instance_id", owner_instance_id_) !=
            FLOW_FUNC_SUCCESS ||
        params->GetAttr("quantum_delay_us", quantum_delay_us_) !=
            FLOW_FUNC_SUCCESS ||
        owner_instance_id_ <= 0 || quantum_delay_us_ < 0 ||
        quantum_delay_us_ > 1000000) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    ResetState();
    FLOW_FUNC_LOG_INFO("persistent P0 owner initialized owner=%ld delay_us=%ld",
                       owner_instance_id_, quantum_delay_us_);
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
      const int32_t timeout = active_ ? 0 : -1;
      const int32_t dequeue = queues[0]->Dequeue(message, timeout);
      if (dequeue == FLOW_FUNC_SUCCESS) {
        Event event;
        if (!ReadEvent(message, event)) {
          if (Emit(context, kOutputRejected, last_event_seq_, generation_,
                   commit_seq_, state_, remaining_, kStatusBadMessage) !=
              FLOW_FUNC_SUCCESS) {
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

      if (!active_) {
        continue;
      }
      if (quantum_delay_us_ > 0) {
        usleep(static_cast<useconds_t>(quantum_delay_us_));
      }
      ++commit_seq_;
      ++total_quanta_;
      ++state_;
      --remaining_;
      if (Emit(context, kOutputCommit, last_event_seq_, generation_, commit_seq_,
               state_, remaining_, kStatusOk) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
      }
      if (remaining_ == 0) {
        active_ = false;
        if (Emit(context, kOutputRetireComplete, last_event_seq_, generation_,
                 commit_seq_, state_, remaining_, kStatusOk) !=
                FLOW_FUNC_SUCCESS ||
            EmitQuiescent(context) != FLOW_FUNC_SUCCESS) {
          return FLOW_FUNC_FAILED;
        }
      }
    }
  }

 private:
  void ResetState() {
    active_ = false;
    generation_ = 0;
    last_generation_ = 0;
    last_event_seq_ = 0;
    last_cancel_seq_ = 0;
    commit_seq_ = 0;
    state_ = 0;
    remaining_ = 0;
    total_quanta_ = 0;
    event_count_ = 0;
    quiescent_count_ = 0;
  }

  int32_t Emit(const std::shared_ptr<MetaRunContext> &context,
               int64_t output_type, int64_t event_seq, int64_t generation,
               int64_t commit_seq, int64_t state, int64_t remaining,
               int64_t status, bool eos = false) {
    auto output = context->AllocTensorMsg(
        {static_cast<int64_t>(kOutputElements)}, TensorDataType::DT_INT64);
    if (output == nullptr || output->GetTensor() == nullptr ||
        output->GetTensor()->GetData() == nullptr) {
      return FLOW_FUNC_FAILED;
    }
    const int64_t values[kOutputElements] = {
        owner_instance_id_, output_type, event_seq, generation,
        commit_seq, state, remaining, status};
    std::memcpy(output->GetTensor()->GetData(), values, sizeof(values));
    output->SetTransactionId(static_cast<uint64_t>(event_seq));
    if (eos) {
      output->SetFlowFlags(static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS));
    }
    return context->SetOutput(0, output);
  }

  int32_t EmitQuiescent(const std::shared_ptr<MetaRunContext> &context) {
    ++quiescent_count_;
    return Emit(context, kOutputQuiescent, last_event_seq_, generation_,
                commit_seq_, state_, remaining_, quiescent_count_);
  }

  int32_t Reject(const std::shared_ptr<MetaRunContext> &context,
                 const Event &event, Status status) {
    return Emit(context, kOutputRejected, event.seq, event.generation,
                commit_seq_, state_, remaining_, status);
  }

  int32_t HandleEvent(const std::shared_ptr<MetaRunContext> &context,
                      const Event &event, bool &shutdown) {
    shutdown = false;
    ++event_count_;
    if (event.owner != owner_instance_id_) {
      return Reject(context, event, kStatusWrongOwner);
    }
    if (event.seq <= last_event_seq_) {
      return Reject(context, event, kStatusStaleEvent);
    }
    last_event_seq_ = event.seq;

    if (event.type == kEventAdmit) {
      if (active_) {
        return Reject(context, event, kStatusBusy);
      }
      if (event.generation <= last_generation_) {
        return Reject(context, event, kStatusBadGeneration);
      }
      if (event.target_quanta <= 0 ||
          event.target_quanta > kMaxSyntheticQuanta) {
        return Reject(context, event, kStatusBadTarget);
      }
      active_ = true;
      generation_ = event.generation;
      last_generation_ = event.generation;
      last_cancel_seq_ = 0;
      commit_seq_ = 0;
      state_ = event.seed;
      remaining_ = event.target_quanta;
      return Emit(context, kOutputAdmitAck, event.seq, generation_, commit_seq_,
                  state_, remaining_, kStatusOk);
    }

    if (event.type == kEventCancel) {
      if (!active_ || event.generation != generation_ ||
          event.cancel_seq <= last_cancel_seq_) {
        return Reject(context, event, kStatusStaleCancel);
      }
      last_cancel_seq_ = event.cancel_seq;
      active_ = false;
      if (Emit(context, kOutputRetireCancelled, event.seq, generation_,
               commit_seq_, state_, remaining_, event.cancel_seq) !=
              FLOW_FUNC_SUCCESS ||
          EmitQuiescent(context) != FLOW_FUNC_SUCCESS) {
        return FLOW_FUNC_FAILED;
      }
      return FLOW_FUNC_SUCCESS;
    }

    if (event.type == kEventShutdown) {
      if (active_) {
        return Reject(context, event, kStatusBusy);
      }
      shutdown = true;
      return Emit(context, kOutputShutdown, event.seq, generation_,
                  total_quanta_, event_count_, quiescent_count_, kStatusOk,
                  true);
    }
    return Reject(context, event, kStatusUnknownEvent);
  }

  int64_t owner_instance_id_ = 0;
  int64_t quantum_delay_us_ = 0;
  bool active_ = false;
  int64_t generation_ = 0;
  int64_t last_generation_ = 0;
  int64_t last_event_seq_ = 0;
  int64_t last_cancel_seq_ = 0;
  int64_t commit_seq_ = 0;
  int64_t state_ = 0;
  int64_t remaining_ = 0;
  int64_t total_quanta_ = 0;
  int64_t event_count_ = 0;
  int64_t quiescent_count_ = 0;
};

FLOW_FUNC_REGISTRAR(PersistentControlP0)
    .RegProcFunc("persistent_control_p0", &PersistentControlP0::Proc);
}  // namespace FlowFunc
