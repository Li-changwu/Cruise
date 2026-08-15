#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "flow_func/meta_multi_func.h"

namespace FlowFunc {
namespace {
constexpr int64_t kResultElements = 19;
constexpr int64_t kApiTensorMsg = 1;
constexpr int64_t kApiFactoryWrap = 2;
constexpr int64_t kApiRawMsg = 3;
constexpr int64_t kApiTensorList = 4;
constexpr int64_t kNotApplicable = -1;
constexpr uint8_t kFirstPattern = 0x5a;
constexpr uint8_t kLastPattern = 0xa5;

const std::vector<int64_t> kShape21MiB = {28, 6, 128, 4, 128};
const std::vector<int64_t> kShape28MiB = {28, 8, 128, 4, 128};
const std::vector<int64_t> kShape42MiB = {28, 12, 128, 4, 128};

enum ProbeStatus : int64_t {
  kProbeOk = 0,
  kProbeUnknownCase = 1,
  kProbeAllocationNull = 2,
  kProbeConversionNull = 3,
  kProbeMessageInvalid = 4,
  kProbeShapeMismatch = 5,
  kProbeDtypeMismatch = 6,
  kProbeElementMismatch = 7,
  kProbeSizeMismatch = 8,
  kProbeDataNull = 9,
  kProbeTouchMismatch = 10,
};

struct Observation {
  int64_t case_id = 0;
  int64_t api_id = 0;
  int64_t requested_bytes = 0;
  int64_t status = kProbeOk;
  int64_t allocated = 0;
  int64_t converted = kNotApplicable;
  int64_t ret_code = kNotApplicable;
  int64_t msg_type = kNotApplicable;
  int64_t tensor_count = 0;
  int64_t shape_ok = 0;
  int64_t dtype_ok = 0;
  int64_t elements = 0;
  int64_t data_bytes = 0;
  int64_t data_nonnull = 0;
  int64_t touch_ok = 0;
  int64_t first = kNotApplicable;
  int64_t last = kNotApplicable;
};

int64_t Elements(const std::vector<int64_t> &shape) {
  int64_t elements = 1;
  for (int64_t extent : shape) elements *= extent;
  return elements;
}

int64_t Bytes(const std::vector<int64_t> &shape) {
  return Elements(shape) * 2;
}

bool Touch(void *data, int64_t bytes, Observation &observation) {
  if (data == nullptr || bytes < 2) return false;
  auto *values = static_cast<volatile uint8_t *>(data);
  values[0] = kFirstPattern;
  values[bytes - 1] = kLastPattern;
  observation.first = values[0];
  observation.last = values[bytes - 1];
  observation.touch_ok = observation.first == kFirstPattern &&
                         observation.last == kLastPattern;
  return observation.touch_ok != 0;
}

void ObserveMessage(const std::shared_ptr<FlowMsg> &message,
                    MsgType expected_type, Observation &observation) {
  if (message == nullptr) {
    observation.status = kProbeAllocationNull;
    return;
  }
  observation.allocated = 1;
  observation.ret_code = message->GetRetCode();
  observation.msg_type = static_cast<int64_t>(message->GetMsgType());
  if (observation.ret_code != FLOW_FUNC_SUCCESS ||
      message->GetMsgType() != expected_type) {
    observation.status = kProbeMessageInvalid;
  }
}

void ObserveTensor(Tensor *tensor, const std::vector<int64_t> &shape,
                   Observation &observation) {
  if (tensor == nullptr) {
    observation.status = kProbeAllocationNull;
    return;
  }
  observation.tensor_count = 1;
  observation.shape_ok = tensor->GetShape() == shape;
  observation.dtype_ok = tensor->GetDataType() == TensorDataType::DT_BF16;
  observation.elements = tensor->GetElementCnt();
  observation.data_bytes = static_cast<int64_t>(tensor->GetDataSize());
  observation.data_nonnull = tensor->GetData() != nullptr;
  if (!observation.shape_ok) {
    observation.status = kProbeShapeMismatch;
  } else if (!observation.dtype_ok) {
    observation.status = kProbeDtypeMismatch;
  } else if (observation.elements != Elements(shape)) {
    observation.status = kProbeElementMismatch;
  } else if (observation.data_bytes != Bytes(shape)) {
    observation.status = kProbeSizeMismatch;
  } else if (!observation.data_nonnull) {
    observation.status = kProbeDataNull;
  } else if (!Touch(tensor->GetData(), observation.data_bytes, observation)) {
    observation.status = kProbeTouchMismatch;
  }
}

Observation ProbeTensorMsg(const std::shared_ptr<MetaRunContext> &context,
                           int64_t case_id,
                           const std::vector<int64_t> &shape) {
  Observation observation;
  observation.case_id = case_id;
  observation.api_id = kApiTensorMsg;
  observation.requested_bytes = Bytes(shape);
  auto message = context->AllocTensorMsg(shape, TensorDataType::DT_BF16);
  ObserveMessage(message, MsgType::MSG_TYPE_TENSOR_DATA, observation);
  if (observation.status == kProbeOk) {
    ObserveTensor(message->GetTensor(), shape, observation);
  }
  return observation;
}

Observation ProbeFactoryWrap(const std::shared_ptr<MetaRunContext> &context,
                             int64_t case_id,
                             const std::vector<int64_t> &shape) {
  Observation observation;
  observation.case_id = case_id;
  observation.api_id = kApiFactoryWrap;
  observation.requested_bytes = Bytes(shape);
  auto tensor = FlowBufferFactory::AllocTensor(shape, TensorDataType::DT_BF16);
  observation.allocated = tensor != nullptr;
  if (tensor == nullptr) {
    observation.status = kProbeAllocationNull;
    return observation;
  }
  ObserveTensor(tensor.get(), shape, observation);
  if (observation.status != kProbeOk) return observation;
  auto message = context->ToFlowMsg(tensor);
  observation.converted = message != nullptr;
  if (message == nullptr) {
    observation.status = kProbeConversionNull;
    return observation;
  }
  ObserveMessage(message, MsgType::MSG_TYPE_TENSOR_DATA, observation);
  if (observation.status == kProbeOk) {
    ObserveTensor(message->GetTensor(), shape, observation);
  }
  return observation;
}

Observation ProbeRawMsg(const std::shared_ptr<MetaRunContext> &context,
                        int64_t case_id, int64_t bytes) {
  Observation observation;
  observation.case_id = case_id;
  observation.api_id = kApiRawMsg;
  observation.requested_bytes = bytes;
  auto message = context->AllocRawDataMsg(bytes);
  ObserveMessage(message, MsgType::MSG_TYPE_RAW_MSG, observation);
  if (observation.status != kProbeOk) return observation;
  void *data = nullptr;
  uint64_t data_size = 0;
  if (message->GetRawData(data, data_size) != FLOW_FUNC_SUCCESS) {
    observation.status = kProbeMessageInvalid;
    return observation;
  }
  observation.data_bytes = static_cast<int64_t>(data_size);
  observation.data_nonnull = data != nullptr;
  if (observation.data_bytes != bytes) {
    observation.status = kProbeSizeMismatch;
  } else if (!observation.data_nonnull) {
    observation.status = kProbeDataNull;
  } else if (!Touch(data, bytes, observation)) {
    observation.status = kProbeTouchMismatch;
  }
  return observation;
}

Observation ProbeTensorMsgGroup(
    const std::shared_ptr<MetaRunContext> &context, int64_t case_id,
    const std::vector<int64_t> &shape, size_t count) {
  Observation observation;
  observation.case_id = case_id;
  observation.api_id = kApiTensorMsg;
  observation.requested_bytes = static_cast<int64_t>(count) * Bytes(shape);
  observation.ret_code = FLOW_FUNC_SUCCESS;
  observation.msg_type = static_cast<int64_t>(MsgType::MSG_TYPE_TENSOR_DATA);
  observation.shape_ok = 1;
  observation.dtype_ok = 1;
  observation.data_nonnull = 1;
  observation.touch_ok = 1;
  std::vector<std::shared_ptr<FlowMsg>> messages;
  messages.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    auto message = context->AllocTensorMsg(shape, TensorDataType::DT_BF16);
    if (message == nullptr) {
      observation.status = kProbeAllocationNull;
      return observation;
    }
    observation.allocated += 1;
    if (message->GetRetCode() != FLOW_FUNC_SUCCESS ||
        message->GetMsgType() != MsgType::MSG_TYPE_TENSOR_DATA) {
      observation.ret_code = message->GetRetCode();
      observation.msg_type = static_cast<int64_t>(message->GetMsgType());
      observation.status = kProbeMessageInvalid;
      return observation;
    }
    Observation item;
    ObserveTensor(message->GetTensor(), shape, item);
    observation.tensor_count += item.tensor_count;
    observation.shape_ok = observation.shape_ok && item.shape_ok;
    observation.dtype_ok = observation.dtype_ok && item.dtype_ok;
    observation.elements += item.elements;
    observation.data_bytes += item.data_bytes;
    observation.data_nonnull = observation.data_nonnull && item.data_nonnull;
    observation.touch_ok = observation.touch_ok && item.touch_ok;
    observation.first = item.first;
    observation.last = item.last;
    if (item.status != kProbeOk) {
      observation.status = item.status;
      return observation;
    }
    messages.push_back(std::move(message));
  }
  return observation;
}

Observation ProbeFactoryWrapGroup(
    const std::shared_ptr<MetaRunContext> &context, int64_t case_id,
    const std::vector<int64_t> &shape, size_t count) {
  Observation observation;
  observation.case_id = case_id;
  observation.api_id = kApiFactoryWrap;
  observation.requested_bytes = static_cast<int64_t>(count) * Bytes(shape);
  observation.ret_code = FLOW_FUNC_SUCCESS;
  observation.msg_type = static_cast<int64_t>(MsgType::MSG_TYPE_TENSOR_DATA);
  observation.shape_ok = 1;
  observation.dtype_ok = 1;
  observation.data_nonnull = 1;
  observation.touch_ok = 1;
  observation.converted = 0;
  std::vector<std::shared_ptr<Tensor>> tensors;
  std::vector<std::shared_ptr<FlowMsg>> messages;
  tensors.reserve(count);
  messages.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    auto tensor = FlowBufferFactory::AllocTensor(shape, TensorDataType::DT_BF16);
    if (tensor == nullptr) {
      observation.status = kProbeAllocationNull;
      return observation;
    }
    observation.allocated += 1;
    Observation before_wrap;
    ObserveTensor(tensor.get(), shape, before_wrap);
    if (before_wrap.status != kProbeOk) {
      observation.status = before_wrap.status;
      return observation;
    }
    auto message = context->ToFlowMsg(tensor);
    if (message == nullptr) {
      observation.status = kProbeConversionNull;
      return observation;
    }
    observation.converted += 1;
    if (message->GetRetCode() != FLOW_FUNC_SUCCESS ||
        message->GetMsgType() != MsgType::MSG_TYPE_TENSOR_DATA) {
      observation.ret_code = message->GetRetCode();
      observation.msg_type = static_cast<int64_t>(message->GetMsgType());
      observation.status = kProbeMessageInvalid;
      return observation;
    }
    Observation item;
    ObserveTensor(message->GetTensor(), shape, item);
    observation.tensor_count += item.tensor_count;
    observation.shape_ok = observation.shape_ok && item.shape_ok;
    observation.dtype_ok = observation.dtype_ok && item.dtype_ok;
    observation.elements += item.elements;
    observation.data_bytes += item.data_bytes;
    observation.data_nonnull = observation.data_nonnull && item.data_nonnull;
    observation.touch_ok = observation.touch_ok && item.touch_ok;
    observation.first = item.first;
    observation.last = item.last;
    if (item.status != kProbeOk) {
      observation.status = item.status;
      return observation;
    }
    tensors.push_back(std::move(tensor));
    messages.push_back(std::move(message));
  }
  return observation;
}

Observation ProbeTensorList(const std::shared_ptr<MetaRunContext> &context,
                            int64_t case_id, size_t count) {
  Observation observation;
  observation.case_id = case_id;
  observation.api_id = kApiTensorList;
  observation.requested_bytes = static_cast<int64_t>(count) * Bytes(kShape21MiB);
  std::vector<std::vector<int64_t>> shapes(count, kShape21MiB);
  std::vector<TensorDataType> data_types(count, TensorDataType::DT_BF16);
  auto message = context->AllocTensorListMsg(shapes, data_types);
  ObserveMessage(message, MsgType::MSG_TYPE_TENSOR_LIST, observation);
  if (observation.status != kProbeOk) return observation;
  const auto tensors = message->GetTensorList();
  observation.tensor_count = static_cast<int64_t>(tensors.size());
  observation.shape_ok = tensors.size() == count;
  observation.dtype_ok = tensors.size() == count;
  observation.elements = 0;
  observation.data_bytes = 0;
  observation.data_nonnull = tensors.size() == count;
  observation.touch_ok = tensors.size() == count;
  for (Tensor *tensor : tensors) {
    if (tensor == nullptr) {
      observation.data_nonnull = 0;
      observation.status = kProbeDataNull;
      continue;
    }
    observation.shape_ok = observation.shape_ok &&
                           tensor->GetShape() == kShape21MiB;
    observation.dtype_ok = observation.dtype_ok &&
                           tensor->GetDataType() == TensorDataType::DT_BF16;
    observation.elements += tensor->GetElementCnt();
    observation.data_bytes += static_cast<int64_t>(tensor->GetDataSize());
    observation.data_nonnull = observation.data_nonnull &&
                               tensor->GetData() != nullptr;
    Observation touched;
    observation.touch_ok = observation.touch_ok &&
        Touch(tensor->GetData(), static_cast<int64_t>(tensor->GetDataSize()),
              touched);
    observation.first = touched.first;
    observation.last = touched.last;
  }
  if (!observation.shape_ok) {
    observation.status = kProbeShapeMismatch;
  } else if (!observation.dtype_ok) {
    observation.status = kProbeDtypeMismatch;
  } else if (observation.elements !=
             static_cast<int64_t>(count) * Elements(kShape21MiB)) {
    observation.status = kProbeElementMismatch;
  } else if (observation.data_bytes != observation.requested_bytes) {
    observation.status = kProbeSizeMismatch;
  } else if (!observation.data_nonnull) {
    observation.status = kProbeDataNull;
  } else if (!observation.touch_ok) {
    observation.status = kProbeTouchMismatch;
  }
  return observation;
}
}  // namespace

class P3AllocatorProbe : public MetaMultiFunc {
 public:
  int32_t Proc(const std::shared_ptr<MetaRunContext> &context,
               const std::vector<std::shared_ptr<FlowMsg>> &inputs) {
    if (context == nullptr || inputs.size() != 1 || inputs[0] == nullptr ||
        inputs[0]->GetRetCode() != FLOW_FUNC_SUCCESS ||
        inputs[0]->GetMsgType() != MsgType::MSG_TYPE_TENSOR_DATA ||
        inputs[0]->GetTensor() == nullptr ||
        inputs[0]->GetTensor()->GetDataType() != TensorDataType::DT_INT64 ||
        inputs[0]->GetTensor()->GetElementCnt() != 1 ||
        inputs[0]->GetTensor()->GetData() == nullptr) {
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    const int64_t case_id =
        *static_cast<const int64_t *>(inputs[0]->GetTensor()->GetData());
    Observation observation;
    if (case_id == 1) {
      observation = ProbeTensorMsg(context, case_id, kShape21MiB);
    } else if (case_id == 2) {
      observation = ProbeTensorMsg(context, case_id, kShape28MiB);
    } else if (case_id == 3) {
      observation = ProbeTensorMsg(context, case_id, kShape42MiB);
    } else if (case_id == 4) {
      observation = ProbeFactoryWrap(context, case_id, kShape28MiB);
    } else if (case_id == 5) {
      observation = ProbeFactoryWrap(context, case_id, kShape42MiB);
    } else if (case_id == 6) {
      observation = ProbeRawMsg(context, case_id, Bytes(kShape28MiB));
    } else if (case_id == 7) {
      observation = ProbeRawMsg(context, case_id, Bytes(kShape42MiB));
    } else if (case_id == 8) {
      observation = ProbeTensorList(context, case_id, 2);
    } else if (case_id == 9) {
      observation = ProbeTensorMsgGroup(context, case_id, kShape42MiB, 2);
    } else if (case_id == 10) {
      observation = ProbeFactoryWrapGroup(context, case_id, kShape42MiB, 2);
    } else if (case_id == 11) {
      observation = ProbeTensorMsgGroup(context, case_id, kShape21MiB, 4);
    } else if (case_id == 12) {
      observation = ProbeTensorList(context, case_id, 4);
    } else {
      observation.case_id = case_id;
      observation.status = kProbeUnknownCase;
    }
    auto output = context->AllocTensorMsg({kResultElements},
                                          TensorDataType::DT_INT64);
    if (output == nullptr || output->GetTensor() == nullptr ||
        output->GetTensor()->GetData() == nullptr ||
        output->GetTensor()->GetElementCnt() != kResultElements) {
      return FLOW_FUNC_FAILED;
    }
    const int64_t values[kResultElements] = {
        observation.case_id, observation.api_id, observation.requested_bytes,
        observation.status, observation.allocated, observation.converted,
        observation.ret_code, observation.msg_type, observation.tensor_count,
        observation.shape_ok, observation.dtype_ok, observation.elements,
        observation.data_bytes, observation.data_nonnull, observation.touch_ok,
        observation.first, observation.last, kFirstPattern, kLastPattern};
    std::memcpy(output->GetTensor()->GetData(), values, sizeof(values));
    return context->SetOutput(0, output);
  }
};

FLOW_FUNC_REGISTRAR(P3AllocatorProbe)
    .RegProcFunc("p3_allocator_probe", &P3AllocatorProbe::Proc);
}  // namespace FlowFunc
