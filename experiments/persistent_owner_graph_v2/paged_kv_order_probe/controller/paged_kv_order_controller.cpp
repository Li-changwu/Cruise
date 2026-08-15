#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "flow_func/meta_multi_func.h"

namespace FlowFunc {
namespace {
constexpr int64_t kBatchSize = 4;
constexpr int64_t kBlocksPerRow = 3;
constexpr int64_t kPhysicalBlocks = 12;
constexpr int64_t kPackedChannels = 32;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kPackWidth = 16;
constexpr int64_t kFeatures = kPackedChannels * kPackWidth;
constexpr int64_t kStateElements =
    2 * kPhysicalBlocks * kPackedChannels * kBlockSize * kPackWidth;
constexpr int64_t kMetadataElements = 5;
constexpr int64_t kReportElements = 18;
constexpr int64_t kSummaryElements = 24;
constexpr int64_t kUpdatedElements = 2 * kBatchSize * kFeatures;
constexpr int32_t kRunModelTimeoutMs = 120000;
constexpr uint32_t kChecksumModulus = 65521U;
constexpr std::array<int32_t, kBatchSize> kPositions = {0, 127, 128, 383};

bool IsTensor(const std::shared_ptr<FlowMsg> &message, TensorDataType type,
              int64_t elements) {
  return message != nullptr && message->GetRetCode() == FLOW_FUNC_SUCCESS &&
         message->GetMsgType() == MsgType::MSG_TYPE_TENSOR_DATA &&
         message->GetTensor() != nullptr &&
         message->GetTensor()->GetDataType() == type &&
         message->GetTensor()->GetElementCnt() == elements &&
         message->GetTensor()->GetData() != nullptr;
}

uint16_t ExpectedBits(int32_t sequence, int64_t cache_kind, int64_t row,
                      int64_t feature) {
  return static_cast<uint16_t>(0x2000U +
                               static_cast<uint32_t>(sequence) * 0x1000U +
                               static_cast<uint32_t>(cache_kind) * 0x0800U +
                               static_cast<uint32_t>(row) * 0x0200U +
                               static_cast<uint32_t>(feature));
}

int64_t StateIndex(int64_t cache_kind, int64_t row, int64_t position,
                   int64_t feature) {
  const int64_t block = row * kBlocksPerRow + position / kBlockSize;
  const int64_t offset = position % kBlockSize;
  const int64_t channel = feature / kPackWidth;
  const int64_t lane = feature % kPackWidth;
  return ((((cache_kind * kPhysicalBlocks + block) * kPackedChannels +
            channel) *
               kBlockSize +
           offset) *
              kPackWidth +
          lane);
}

uint32_t UpdatedChecksum(int32_t sequence) {
  uint32_t checksum = 0;
  for (int64_t cache_kind = 0; cache_kind < 2; ++cache_kind) {
    for (int64_t row = 0; row < kBatchSize; ++row) {
      for (int64_t feature = 0; feature < kFeatures; ++feature) {
        checksum =
            (checksum + ExpectedBits(sequence, cache_kind, row, feature)) %
            kChecksumModulus;
      }
    }
  }
  return checksum;
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

bool FullStateExact(const uint16_t *state, int32_t sequence) {
  for (int64_t index = 0; index < kStateElements; ++index) {
    int64_t remainder = index;
    const int64_t lane = remainder % kPackWidth;
    remainder /= kPackWidth;
    const int64_t offset = remainder % kBlockSize;
    remainder /= kBlockSize;
    const int64_t channel = remainder % kPackedChannels;
    remainder /= kPackedChannels;
    const int64_t block = remainder % kPhysicalBlocks;
    const int64_t cache_kind = remainder / kPhysicalBlocks;
    const int64_t row = block / kBlocksPerRow;
    const int64_t page = block % kBlocksPerRow;
    const int64_t position = page * kBlockSize + offset;
    const int64_t feature = channel * kPackWidth + lane;
    const uint16_t expected =
        position == kPositions[row]
            ? ExpectedBits(sequence, cache_kind, row, feature)
            : 0U;
    if (state[index] != expected) return false;
  }
  return true;
}

bool ReportExact(const std::array<int32_t, kReportElements> &report,
                 int32_t sequence) {
  const uint32_t checksum = UpdatedChecksum(sequence);
  return report[0] == sequence && report[1] == sequence && report[2] == 0 &&
         report[3] == kUpdatedElements && report[4] == kUpdatedElements &&
         report[5] == 0 && report[6] == static_cast<int32_t>(checksum) &&
         report[7] == static_cast<int32_t>(checksum) && report[8] == 0 &&
         report[9] == 511 && report[10] == 896 && report[11] == 1535 &&
         report[12] == ExpectedBits(sequence, 0, 0, 0) &&
         report[13] == ExpectedBits(sequence, 0, 3, kFeatures - 1) &&
         report[14] == ExpectedBits(sequence, 1, 0, 0) &&
         report[15] == ExpectedBits(sequence, 1, 3, kFeatures - 1) &&
         report[16] == 0 && report[17] == 0;
}
}  // namespace

class PagedKvOrderController : public MetaMultiFunc {
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
    if ((sequence != 1 && sequence != 2) || sequence != graph_calls_ + 1) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    for (int64_t row = 0; row < kBatchSize; ++row) {
      if (metadata[row + 1] != kPositions[row]) {
        return FLOW_FUNC_ERR_PARAM_INVALID;
      }
    }
    if (EnsureState(context) != FLOW_FUNC_SUCCESS) return FLOW_FUNC_FAILED;

    auto *state =
        static_cast<uint16_t *>(state_message_->GetTensor()->GetData());
    const bool pointer_stable = state == state_address_;
    const bool flowmsg_stable = state_message_.get() == state_message_address_;
    const uint16_t direct_before = state[StateIndex(0, 0, kPositions[0], 0)];
    const uint16_t expected_before =
        sequence == 1 ? 0U : ExpectedBits(sequence - 1, 0, 0, 0);
    const uint32_t checksum_before = Adler32(state, kStateElements);

    std::vector<std::shared_ptr<FlowMsg>> outputs;
    const int32_t model_status = context->RunFlowModel(
        "paged_kv_order_graph_0", {state_message_, inputs[0]}, outputs,
        kRunModelTimeoutMs);
    ++graph_calls_;

    std::array<int32_t, kReportElements> report{};
    bool report_exact = false;
    if (model_status == FLOW_FUNC_SUCCESS && outputs.size() == 1 &&
        IsTensor(outputs[0], TensorDataType::DT_INT32, kReportElements)) {
      std::memcpy(report.data(), outputs[0]->GetTensor()->GetData(),
                  sizeof(report));
      report_exact = ReportExact(report, sequence);
    }

    const uint16_t direct_after = state[StateIndex(0, 0, kPositions[0], 0)];
    const uint32_t checksum_after = Adler32(state, kStateElements);
    const bool full_state_exact = FullStateExact(state, sequence);
    const bool pass = pointer_stable && flowmsg_stable &&
                      direct_before == expected_before &&
                      direct_after == ExpectedBits(sequence, 0, 0, 0) &&
                      report_exact && full_state_exact;

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
        full_state_exact ? 1 : 0,
        kStateElements,
        kStateElements * static_cast<int64_t>(sizeof(uint16_t)),
        0,
        0,
        kReportElements,
        checksum_before,
        checksum_after,
        direct_before,
        direct_after,
        report[5],
        report[6],
        report[7],
        report[16],
        report[17],
        UpdatedChecksum(sequence),
    };
    std::memcpy(summary->GetTensor()->GetData(), values, sizeof(values));
    return context->SetOutput(0, summary);
  }

 private:
  int32_t EnsureState(const std::shared_ptr<MetaRunContext> &context) {
    if (state_message_ != nullptr) {
      return IsTensor(state_message_, TensorDataType::DT_BF16, kStateElements)
                 ? FLOW_FUNC_SUCCESS
                 : FLOW_FUNC_FAILED;
    }
    auto tensor = FlowBufferFactory::AllocTensor(
        {2, kPhysicalBlocks, kPackedChannels, kBlockSize, kPackWidth},
        TensorDataType::DT_BF16);
    if (tensor == nullptr) return FLOW_FUNC_FAILED;
    state_message_ = context->ToFlowMsg(tensor);
    if (!IsTensor(state_message_, TensorDataType::DT_BF16, kStateElements)) {
      return FLOW_FUNC_FAILED;
    }
    auto *state =
        static_cast<uint16_t *>(state_message_->GetTensor()->GetData());
    std::memset(state, 0, state_message_->GetTensor()->GetDataSize());
    state_address_ = state;
    state_message_address_ = state_message_.get();
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
    state_message_.reset();
    state_address_ = nullptr;
    state_message_address_ = nullptr;
    allocation_count_ = 0;
    graph_calls_ = 0;
    return FLOW_FUNC_SUCCESS;
  }

 private:
  std::shared_ptr<FlowMsg> state_message_;
  void *state_address_ = nullptr;
  FlowMsg *state_message_address_ = nullptr;
  int64_t allocation_count_ = 0;
  int64_t graph_calls_ = 0;
};

FLOW_FUNC_REGISTRAR(PagedKvOrderController)
    .RegProcFunc("paged_kv_order_controller", &PagedKvOrderController::Proc);
}  // namespace FlowFunc
