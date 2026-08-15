#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "flow_func/meta_multi_func.h"

namespace FlowFunc {
namespace {
constexpr int64_t kCacheElements = 4096;
constexpr int64_t kMetadataElements = 4;
constexpr int64_t kReportElements = 9;
constexpr int64_t kSummaryElements = 22;
constexpr int32_t kRunModelTimeoutMs = 120000;
constexpr int32_t kSlot = 2048;
constexpr uint16_t kLeftSentinel = 0x3e00U;
constexpr uint16_t kRightSentinel = 0xbe00U;

bool IsTensor(const std::shared_ptr<FlowMsg> &message, TensorDataType type,
              int64_t elements) {
  return message != nullptr && message->GetRetCode() == FLOW_FUNC_SUCCESS &&
         message->GetMsgType() == MsgType::MSG_TYPE_TENSOR_DATA &&
         message->GetTensor() != nullptr &&
         message->GetTensor()->GetDataType() == type &&
         message->GetTensor()->GetElementCnt() == elements &&
         message->GetTensor()->GetData() != nullptr;
}

uint32_t Adler32(const uint16_t *values, size_t elements) {
  constexpr uint32_t kModulus = 65521U;
  uint32_t first = 1U;
  uint32_t second = 0U;
  const auto *bytes = reinterpret_cast<const uint8_t *>(values);
  size_t remaining = elements * sizeof(uint16_t);
  while (remaining > 0) {
    const size_t chunk = remaining > 5552 ? 5552 : remaining;
    for (size_t index = 0; index < chunk; ++index) {
      first += bytes[index];
      second += first;
    }
    first %= kModulus;
    second %= kModulus;
    bytes += chunk;
    remaining -= chunk;
  }
  return (second << 16) | first;
}
}  // namespace

class DeviceKvUpdateController : public MetaMultiFunc {
 public:
  int32_t Proc(const std::shared_ptr<MetaRunContext> &context,
               const std::vector<std::shared_ptr<FlowMsg>> &inputs) {
    if (context == nullptr || inputs.size() != 1 ||
        !IsTensor(inputs[0], TensorDataType::DT_INT32, kMetadataElements)) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    const auto *metadata =
        static_cast<const int32_t *>(inputs[0]->GetTensor()->GetData());
    const int32_t sequence = metadata[0];
    const int32_t slot = metadata[1];
    const uint16_t expected_before =
        static_cast<uint16_t>(metadata[2] & 0xffff);
    const uint16_t requested_new =
        static_cast<uint16_t>(metadata[3] & 0xffff);
    if ((sequence != 1 && sequence != 2) || slot != kSlot ||
        sequence != graph_calls_ + 1) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (EnsureCache(context) != FLOW_FUNC_SUCCESS) return FLOW_FUNC_FAILED;

    auto *cache =
        static_cast<uint16_t *>(state_cache_->GetTensor()->GetData());
    const bool pointer_stable = cache == cache_address_;
    const bool flowmsg_stable = state_cache_.get() == cache_message_address_;
    const uint16_t direct_before = cache[slot];
    const uint32_t checksum_before = Adler32(cache, kCacheElements);

    std::vector<std::shared_ptr<FlowMsg>> outputs;
    const int32_t model_status = context->RunFlowModel(
        "kv_update_graph_0", {state_cache_, inputs[0]}, outputs,
        kRunModelTimeoutMs);
    ++graph_calls_;

    bool report_exact = false;
    std::array<int32_t, kReportElements> report{};
    if (model_status == FLOW_FUNC_SUCCESS && outputs.size() == 1 &&
        IsTensor(outputs[0], TensorDataType::DT_INT32, kReportElements)) {
      std::memcpy(report.data(), outputs[0]->GetTensor()->GetData(),
                  sizeof(report));
      report_exact = report[0] == sequence && report[1] == slot &&
                     report[2] == expected_before &&
                     report[3] == requested_new &&
                     report[4] == expected_before &&
                     report[5] == requested_new &&
                     report[6] == kLeftSentinel &&
                     report[7] == kRightSentinel && report[8] == 0;
    }

    const uint16_t direct_after = cache[slot];
    const uint32_t checksum_after = Adler32(cache, kCacheElements);
    bool full_cache_exact = true;
    for (int64_t index = 0; index < kCacheElements; ++index) {
      uint16_t expected = 0;
      if (index == kSlot - 1) expected = kLeftSentinel;
      if (index == kSlot) expected = requested_new;
      if (index == kSlot + 1) expected = kRightSentinel;
      if (cache[index] != expected) {
        full_cache_exact = false;
        break;
      }
    }
    const bool pass = pointer_stable && flowmsg_stable &&
                      direct_before == expected_before &&
                      direct_after == requested_new && report_exact &&
                      full_cache_exact;

    auto summary = context->AllocTensorMsg({kSummaryElements},
                                           TensorDataType::DT_INT64);
    if (!IsTensor(summary, TensorDataType::DT_INT64, kSummaryElements)) {
      return FLOW_FUNC_FAILED;
    }
    const int64_t values[kSummaryElements] = {
        sequence,
        pass ? 0 : 1,
        allocation_count_,
        graph_calls_,
        pointer_stable ? 1 : 0,
        flowmsg_stable ? 1 : 0,
        model_status,
        report_exact ? 1 : 0,
        slot,
        direct_before,
        direct_after,
        report[2],
        report[3],
        report[6],
        report[7],
        full_cache_exact ? 1 : 0,
        kCacheElements * static_cast<int64_t>(sizeof(uint16_t)),
        0,
        0,
        kReportElements,
        checksum_before,
        checksum_after,
    };
    std::memcpy(summary->GetTensor()->GetData(), values, sizeof(values));
    return context->SetOutput(0, summary);
  }

 private:
  int32_t EnsureCache(const std::shared_ptr<MetaRunContext> &context) {
    if (state_cache_ != nullptr) {
      return IsTensor(state_cache_, TensorDataType::DT_BF16, kCacheElements)
                 ? FLOW_FUNC_SUCCESS
                 : FLOW_FUNC_FAILED;
    }
    auto tensor = FlowBufferFactory::AllocTensor(
        {kCacheElements}, TensorDataType::DT_BF16);
    if (tensor == nullptr) return FLOW_FUNC_FAILED;
    state_cache_ = context->ToFlowMsg(tensor);
    if (!IsTensor(state_cache_, TensorDataType::DT_BF16, kCacheElements)) {
      return FLOW_FUNC_FAILED;
    }
    auto *cache =
        static_cast<uint16_t *>(state_cache_->GetTensor()->GetData());
    std::memset(cache, 0, state_cache_->GetTensor()->GetDataSize());
    cache[kSlot - 1] = kLeftSentinel;
    cache[kSlot + 1] = kRightSentinel;
    cache_address_ = cache;
    cache_message_address_ = state_cache_.get();
    ++allocation_count_;
    return FLOW_FUNC_SUCCESS;
  }

 public:
  int32_t Init(const std::shared_ptr<MetaParams> &params) override {
    if (params == nullptr || params->GetInputNum() != 1 ||
        params->GetOutputNum() != 1) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    return FLOW_FUNC_SUCCESS;
  }

  int32_t ResetFlowFuncState(const std::shared_ptr<MetaParams> &params) override {
    (void)params;
    state_cache_.reset();
    cache_address_ = nullptr;
    cache_message_address_ = nullptr;
    allocation_count_ = 0;
    graph_calls_ = 0;
    return FLOW_FUNC_SUCCESS;
  }

 private:
  std::shared_ptr<FlowMsg> state_cache_;
  void *cache_address_ = nullptr;
  FlowMsg *cache_message_address_ = nullptr;
  int64_t allocation_count_ = 0;
  int64_t graph_calls_ = 0;
};

FLOW_FUNC_REGISTRAR(DeviceKvUpdateController)
    .RegProcFunc("device_kv_update_controller",
                 &DeviceKvUpdateController::Proc);
}  // namespace FlowFunc
