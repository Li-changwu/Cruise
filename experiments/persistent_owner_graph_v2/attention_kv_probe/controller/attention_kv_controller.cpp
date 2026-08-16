#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "attention_kv_graph_abi.h"
#include "flow_func/meta_multi_func.h"

namespace FlowFunc {
namespace {
namespace GraphAbi = CruiseAttentionKvGraphAbi;
constexpr int64_t kBatch = 4;
constexpr int64_t kBlocksPerRow = 3;
constexpr int64_t kPhysicalBlocks = 12;
constexpr int64_t kKvHeads = 4;
constexpr int64_t kPacksPerHead = 8;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kPackWidth = 16;
constexpr int64_t kHeadDim = kPacksPerHead * kPackWidth;
constexpr int64_t kFeatures = kKvHeads * kHeadDim;
constexpr int64_t kCacheElements =
    kPhysicalBlocks * kKvHeads * kPacksPerHead * kBlockSize * kPackWidth;
constexpr int64_t kQueryElements = kBatch * 28 * kHeadDim;
constexpr int64_t kMaskElements = kBatch * kBlocksPerRow * kBlockSize;
constexpr int64_t kMetadataElements = 5;
constexpr int64_t kTicketElements = 9;
constexpr int64_t kSummaryElements = 32;
constexpr int32_t kRunModelTimeoutMs = 180000;
constexpr std::array<int32_t, kBatch> kPositions = {0, 127, 128, 383};

bool IsTensor(const std::shared_ptr<FlowMsg> &message, TensorDataType type,
              int64_t elements) {
  return message != nullptr && message->GetRetCode() == FLOW_FUNC_SUCCESS &&
         message->GetMsgType() == MsgType::MSG_TYPE_TENSOR_DATA &&
         message->GetTensor() != nullptr &&
         message->GetTensor()->GetDataType() == type &&
         message->GetTensor()->GetElementCnt() == elements &&
         message->GetTensor()->GetData() != nullptr;
}

uint16_t SequenceBits(int32_t sequence) {
  return sequence == 1 ? 0x3f80U : 0x4000U;
}

int64_t CacheIndex(int64_t row, int64_t position, int64_t feature) {
  const int64_t block = row * kBlocksPerRow + position / kBlockSize;
  const int64_t offset = position % kBlockSize;
  const int64_t head = feature / kHeadDim;
  const int64_t dim = feature % kHeadDim;
  const int64_t pack = dim / kPackWidth;
  const int64_t lane = dim % kPackWidth;
  return (((((block * kKvHeads + head) * kPacksPerHead + pack) * kBlockSize +
            offset) *
               kPackWidth) +
          lane);
}

bool FullCacheExact(const uint16_t *cache, int32_t sequence) {
  for (int64_t index = 0; index < kCacheElements; ++index) {
    int64_t remainder = index;
    const int64_t lane = remainder % kPackWidth;
    remainder /= kPackWidth;
    const int64_t offset = remainder % kBlockSize;
    remainder /= kBlockSize;
    const int64_t pack = remainder % kPacksPerHead;
    remainder /= kPacksPerHead;
    const int64_t head = remainder % kKvHeads;
    const int64_t block = remainder / kKvHeads;
    const int64_t row = block / kBlocksPerRow;
    const int64_t page = block % kBlocksPerRow;
    const int64_t position = page * kBlockSize + offset;
    const bool updated = row < kBatch && position == kPositions[row];
    const uint16_t expected =
        updated && sequence > 0 ? SequenceBits(sequence) : 0U;
    (void)lane;
    (void)pack;
    (void)head;
    if (cache[index] != expected) return false;
  }
  return true;
}

bool ExactTicket(const int32_t *ticket, int32_t sequence) {
  return ticket[0] == sequence && ticket[1] == 0 && ticket[2] == 4096 &&
         ticket[3] == 0 && ticket[4] == 127 && ticket[5] == 128 &&
         ticket[6] == 383 && ticket[7] == SequenceBits(sequence) &&
         ticket[8] == SequenceBits(sequence);
}

bool AllBits(const uint16_t *values, int64_t elements, uint16_t expected) {
  for (int64_t index = 0; index < elements; ++index) {
    if (values[index] != expected) return false;
  }
  return true;
}
}  // namespace

class AttentionKvController : public MetaMultiFunc {
 public:
  int32_t Proc(const std::shared_ptr<MetaRunContext> &context,
               const std::vector<std::shared_ptr<FlowMsg>> &inputs) {
    if (context == nullptr || inputs.size() != 1 ||
        !IsTensor(inputs[0], TensorDataType::DT_INT32, kMetadataElements)) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (EnsureState(context) != FLOW_FUNC_SUCCESS) return FLOW_FUNC_FAILED;
    const auto *metadata =
        static_cast<const int32_t *>(inputs[0]->GetTensor()->GetData());
    const int32_t sequence = metadata[0];
    if ((sequence != 1 && sequence != 2) || graph_calls_ + 1 != sequence) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    for (int64_t row = 0; row < kBatch; ++row) {
      if (metadata[row + 1] != kPositions[row]) {
        return FLOW_FUNC_ERR_PARAM_INVALID;
      }
    }

    auto *key = static_cast<uint16_t *>(key_message_->GetTensor()->GetData());
    auto *value = static_cast<uint16_t *>(value_message_->GetTensor()->GetData());
    const bool before_exact =
        FullCacheExact(key, sequence - 1) && FullCacheExact(value, sequence - 1);
    std::vector<std::shared_ptr<FlowMsg>> graph_inputs(GraphAbi::kInputCount);
    graph_inputs[GraphAbi::kKeyCacheInput] = key_message_;
    graph_inputs[GraphAbi::kValueCacheInput] = value_message_;
    graph_inputs[GraphAbi::kMetadataInput] = inputs[0];
    graph_inputs[GraphAbi::kQueryInput] = query_message_;
    graph_inputs[GraphAbi::kMaskInput] = mask_message_;
    graph_inputs[GraphAbi::kBlockTableInput] = block_table_message_;
    std::vector<std::shared_ptr<FlowMsg>> outputs;
    const int32_t model_status = context->RunFlowModel(
        "attention_kv_graph_0", graph_inputs, outputs, kRunModelTimeoutMs);
    ++graph_calls_;

    bool attention_exact = false;
    bool ticket_exact = false;
    uint16_t attention_first = 0;
    uint16_t attention_last = 0;
    int32_t ticket_status = -1;
    int32_t updated_elements = 0;
    if (model_status == FLOW_FUNC_SUCCESS && outputs.size() == 2 &&
        IsTensor(outputs[0], TensorDataType::DT_BF16, kQueryElements) &&
        IsTensor(outputs[1], TensorDataType::DT_INT32, kTicketElements)) {
      const auto *attention =
          static_cast<const uint16_t *>(outputs[0]->GetTensor()->GetData());
      const auto *ticket =
          static_cast<const int32_t *>(outputs[1]->GetTensor()->GetData());
      attention_exact =
          AllBits(attention, kQueryElements, SequenceBits(sequence));
      ticket_exact = ExactTicket(ticket, sequence);
      attention_first = attention[0];
      attention_last = attention[kQueryElements - 1];
      ticket_status = ticket[1];
      updated_elements = ticket[2];
    }
    const bool after_exact =
        FullCacheExact(key, sequence) && FullCacheExact(value, sequence);
    const bool identities_stable =
        key_message_.get() == key_message_address_ &&
        value_message_.get() == value_message_address_;
    const bool addresses_stable = key == key_address_ && value == value_address_;
    const bool mask_exact = MaskExact();
    const bool table_exact = BlockTableExact();
    const bool query_exact = AllBits(
        static_cast<const uint16_t *>(query_message_->GetTensor()->GetData()),
        kQueryElements, 0x3f80U);

    auto summary = context->AllocTensorMsg({kSummaryElements},
                                           TensorDataType::DT_INT64);
    if (!IsTensor(summary, TensorDataType::DT_INT64, kSummaryElements)) {
      return FLOW_FUNC_FAILED;
    }
    const int64_t fields[kSummaryElements] = {
        sequence,
        model_status,
        allocation_count_,
        graph_calls_,
        identities_stable ? 1 : 0,
        addresses_stable ? 1 : 0,
        before_exact ? 1 : 0,
        after_exact ? 1 : 0,
        attention_exact ? 1 : 0,
        ticket_exact ? 1 : 0,
        2 * kCacheElements * static_cast<int64_t>(sizeof(uint16_t)),
        0,
        0,
        kQueryElements,
        kTicketElements,
        sequence == 1 ? 0 : SequenceBits(sequence - 1),
        SequenceBits(sequence),
        SequenceBits(sequence),
        attention_first,
        attention_last,
        ticket_status,
        updated_elements,
        0,
        0,
        6,
        2,
        kMetadataElements * static_cast<int64_t>(sizeof(int32_t)),
        kSummaryElements * static_cast<int64_t>(sizeof(int64_t)),
        before_exact ? 1 : 0,
        mask_exact ? 1 : 0,
        table_exact ? 1 : 0,
        query_exact ? 1 : 0,
    };
    std::memcpy(summary->GetTensor()->GetData(), fields, sizeof(fields));
    return context->SetOutput(0, summary);
  }

  int32_t Init(const std::shared_ptr<MetaParams> &params) override {
    if (params == nullptr || params->GetInputNum() != 1 ||
        params->GetOutputNum() != 1) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    return FLOW_FUNC_SUCCESS;
  }

  int32_t ResetFlowFuncState(const std::shared_ptr<MetaParams> &params) override {
    (void)params;
    key_message_.reset();
    value_message_.reset();
    query_message_.reset();
    mask_message_.reset();
    block_table_message_.reset();
    key_address_ = nullptr;
    value_address_ = nullptr;
    key_message_address_ = nullptr;
    value_message_address_ = nullptr;
    allocation_count_ = 0;
    graph_calls_ = 0;
    return FLOW_FUNC_SUCCESS;
  }

 private:
  std::shared_ptr<FlowMsg> Allocate(const std::shared_ptr<MetaRunContext> &context,
                                    const std::vector<int64_t> &shape,
                                    TensorDataType type) {
    auto tensor = FlowBufferFactory::AllocTensor(shape, type);
    return tensor == nullptr ? nullptr : context->ToFlowMsg(tensor);
  }

  int32_t EnsureState(const std::shared_ptr<MetaRunContext> &context) {
    if (key_message_ != nullptr) {
      return IsTensor(key_message_, TensorDataType::DT_BF16, kCacheElements) &&
                     IsTensor(value_message_, TensorDataType::DT_BF16,
                              kCacheElements)
                 ? FLOW_FUNC_SUCCESS
                 : FLOW_FUNC_FAILED;
    }
    const std::vector<int64_t> cache_shape = {12, 4, 8, 128, 16};
    key_message_ = Allocate(context, cache_shape, TensorDataType::DT_BF16);
    value_message_ = Allocate(context, cache_shape, TensorDataType::DT_BF16);
    query_message_ =
        Allocate(context, {4, 28, 1, 128}, TensorDataType::DT_BF16);
    mask_message_ =
        Allocate(context, {4, 1, 1, 384}, TensorDataType::DT_BOOL);
    block_table_message_ =
        Allocate(context, {4, 3}, TensorDataType::DT_INT32);
    if (!IsTensor(key_message_, TensorDataType::DT_BF16, kCacheElements) ||
        !IsTensor(value_message_, TensorDataType::DT_BF16, kCacheElements) ||
        !IsTensor(query_message_, TensorDataType::DT_BF16, kQueryElements) ||
        !IsTensor(mask_message_, TensorDataType::DT_BOOL, kMaskElements) ||
        !IsTensor(block_table_message_, TensorDataType::DT_INT32, 12)) {
      return FLOW_FUNC_FAILED;
    }
    std::memset(key_message_->GetTensor()->GetData(), 0,
                key_message_->GetTensor()->GetDataSize());
    std::memset(value_message_->GetTensor()->GetData(), 0,
                value_message_->GetTensor()->GetDataSize());
    auto *query =
        static_cast<uint16_t *>(query_message_->GetTensor()->GetData());
    for (int64_t index = 0; index < kQueryElements; ++index) query[index] = 0x3f80U;
    auto *mask = static_cast<uint8_t *>(mask_message_->GetTensor()->GetData());
    std::memset(mask, 1, kMaskElements);
    for (int64_t row = 0; row < kBatch; ++row) {
      mask[row * 384 + kPositions[row]] = 0;
    }
    auto *table =
        static_cast<int32_t *>(block_table_message_->GetTensor()->GetData());
    for (int64_t index = 0; index < 12; ++index) table[index] = index;
    key_address_ = key_message_->GetTensor()->GetData();
    value_address_ = value_message_->GetTensor()->GetData();
    key_message_address_ = key_message_.get();
    value_message_address_ = value_message_.get();
    ++allocation_count_;
    return FLOW_FUNC_SUCCESS;
  }

  bool MaskExact() const {
    const auto *mask =
        static_cast<const uint8_t *>(mask_message_->GetTensor()->GetData());
    for (int64_t row = 0; row < kBatch; ++row) {
      for (int64_t position = 0; position < 384; ++position) {
        const uint8_t expected = position == kPositions[row] ? 0 : 1;
        if (mask[row * 384 + position] != expected) return false;
      }
    }
    return true;
  }

  bool BlockTableExact() const {
    const auto *table = static_cast<const int32_t *>(
        block_table_message_->GetTensor()->GetData());
    for (int64_t index = 0; index < 12; ++index) {
      if (table[index] != index) return false;
    }
    return true;
  }

  std::shared_ptr<FlowMsg> key_message_;
  std::shared_ptr<FlowMsg> value_message_;
  std::shared_ptr<FlowMsg> query_message_;
  std::shared_ptr<FlowMsg> mask_message_;
  std::shared_ptr<FlowMsg> block_table_message_;
  void *key_address_ = nullptr;
  void *value_address_ = nullptr;
  FlowMsg *key_message_address_ = nullptr;
  FlowMsg *value_message_address_ = nullptr;
  int64_t allocation_count_ = 0;
  int64_t graph_calls_ = 0;
};

FLOW_FUNC_REGISTRAR(AttentionKvController)
    .RegProcFunc("attention_kv_controller", &AttentionKvController::Proc);

}  // namespace FlowFunc
