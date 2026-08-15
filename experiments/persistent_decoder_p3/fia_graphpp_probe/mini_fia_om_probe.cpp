#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>

#include "acl/acl.h"
#include "ge/ge_ir_build.h"
#include "graph/graph.h"

namespace {
constexpr int64_t kNotRun = -1;
constexpr size_t kQueryElements = 4 * 28 * 1 * 128;
constexpr size_t kKvElements = 4 * 4 * 384 * 128;
constexpr size_t kMaskElements = 4 * 1 * 1 * 384;
constexpr std::array<size_t, 4> kInputBytes = {
    kQueryElements * sizeof(uint16_t),
    kKvElements * sizeof(uint16_t),
    kKvElements * sizeof(uint16_t),
    kMaskElements,
};
constexpr size_t kOutputBytes = kQueryElements * sizeof(uint16_t);

struct ProbeResult {
  bool pass = false;
  int64_t build_initialize_status = kNotRun;
  int64_t air_load_status = kNotRun;
  int64_t build_status = kNotRun;
  int64_t save_status = kNotRun;
  uint64_t model_buffer_bytes = 0;
  uint64_t om_bytes = 0;
  int64_t acl_init_status = kNotRun;
  int64_t set_device_status = kNotRun;
  int64_t load_status = kNotRun;
  int64_t desc_status = kNotRun;
  uint64_t input_count = 0;
  uint64_t output_count = 0;
  bool io_contract_exact = false;
  int64_t dataset_status = kNotRun;
  int64_t execute_status = kNotRun;
  int64_t output_copy_status = kNotRun;
  bool output_exact = false;
  int64_t resource_cleanup_status = kNotRun;
  int64_t unload_status = kNotRun;
  int64_t reset_device_status = kNotRun;
  int64_t acl_finalize_status = kNotRun;
};

struct DatasetResources {
  aclmdlDataset *dataset = nullptr;
  std::vector<aclDataBuffer *> buffers;
  std::vector<void *> device_buffers;

  int64_t Destroy() {
    int64_t first_error = ACL_SUCCESS;
    for (auto *buffer : buffers) {
      if (buffer == nullptr) continue;
      const auto status = aclDestroyDataBuffer(buffer);
      if (first_error == ACL_SUCCESS && status != ACL_SUCCESS) {
        first_error = status;
      }
    }
    for (auto *device_buffer : device_buffers) {
      if (device_buffer == nullptr) continue;
      const auto status = aclrtFree(device_buffer);
      if (first_error == ACL_SUCCESS && status != ACL_SUCCESS) {
        first_error = status;
      }
    }
    if (dataset != nullptr) {
      const auto status = aclmdlDestroyDataset(dataset);
      if (first_error == ACL_SUCCESS && status != ACL_SUCCESS) {
        first_error = status;
      }
    }
    buffers.clear();
    device_buffers.clear();
    dataset = nullptr;
    return first_error;
  }
};

int64_t StatusValue(ge::graphStatus status) {
  return static_cast<uint32_t>(status);
}

std::vector<uint8_t> ReadBinary(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
}

std::vector<uint8_t> Bf16Ones(size_t bytes) {
  std::vector<uint8_t> values(bytes);
  for (size_t offset = 0; offset + 1 < bytes; offset += 2) {
    values[offset] = 0x80;
    values[offset + 1] = 0x3f;
  }
  return values;
}

int64_t AddDeviceBuffer(DatasetResources *resources, size_t bytes,
                        const uint8_t *host_data) {
  void *device_buffer = nullptr;
  auto status = aclrtMalloc(&device_buffer, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (status != ACL_SUCCESS) return status;
  resources->device_buffers.push_back(device_buffer);
  if (host_data != nullptr) {
    status = aclrtMemcpy(device_buffer, bytes, host_data, bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACL_SUCCESS) return status;
  }
  auto *data_buffer = aclCreateDataBuffer(device_buffer, bytes);
  if (data_buffer == nullptr) return -2;
  resources->buffers.push_back(data_buffer);
  return aclmdlAddDatasetBuffer(resources->dataset, data_buffer);
}

void ExecuteModel(uint32_t model_id, ProbeResult *result) {
  auto *desc = aclmdlCreateDesc();
  if (desc == nullptr) {
    result->desc_status = -2;
    return;
  }
  result->desc_status = aclmdlGetDesc(desc, model_id);
  if (result->desc_status != ACL_SUCCESS) {
    result->resource_cleanup_status = aclmdlDestroyDesc(desc);
    return;
  }

  result->input_count = aclmdlGetNumInputs(desc);
  result->output_count = aclmdlGetNumOutputs(desc);
  result->io_contract_exact = result->input_count == kInputBytes.size() &&
                              result->output_count == 1;
  if (result->io_contract_exact) {
    for (size_t index = 0; index < kInputBytes.size(); ++index) {
      result->io_contract_exact =
          result->io_contract_exact &&
          aclmdlGetInputSizeByIndex(desc, index) == kInputBytes[index];
    }
    result->io_contract_exact =
        result->io_contract_exact &&
        aclmdlGetOutputSizeByIndex(desc, 0) == kOutputBytes;
  }

  DatasetResources inputs;
  DatasetResources outputs;
  inputs.dataset = aclmdlCreateDataset();
  outputs.dataset = aclmdlCreateDataset();
  if (!result->io_contract_exact || inputs.dataset == nullptr ||
      outputs.dataset == nullptr) {
    result->dataset_status = -2;
  } else {
    std::vector<std::vector<uint8_t>> host_inputs = {
        Bf16Ones(kInputBytes[0]),
        Bf16Ones(kInputBytes[1]),
        Bf16Ones(kInputBytes[2]),
        std::vector<uint8_t>(kInputBytes[3], 0),
    };
    result->dataset_status = ACL_SUCCESS;
    for (size_t index = 0;
         index < host_inputs.size() && result->dataset_status == ACL_SUCCESS;
         ++index) {
      result->dataset_status = AddDeviceBuffer(
          &inputs, host_inputs[index].size(), host_inputs[index].data());
    }
    if (result->dataset_status == ACL_SUCCESS) {
      result->dataset_status = AddDeviceBuffer(&outputs, kOutputBytes, nullptr);
    }
    if (result->dataset_status == ACL_SUCCESS) {
      result->execute_status =
          aclmdlExecute(model_id, inputs.dataset, outputs.dataset);
    }
    if (result->execute_status == ACL_SUCCESS) {
      std::vector<uint8_t> output(kOutputBytes);
      result->output_copy_status =
          aclrtMemcpy(output.data(), output.size(), outputs.device_buffers[0],
                      output.size(), ACL_MEMCPY_DEVICE_TO_HOST);
      result->output_exact = result->output_copy_status == ACL_SUCCESS;
      for (size_t offset = 0;
           offset + 1 < output.size() && result->output_exact; offset += 2) {
        result->output_exact =
            output[offset] == 0x80 && output[offset + 1] == 0x3f;
      }
    }
  }

  const auto input_cleanup = inputs.Destroy();
  const auto output_cleanup = outputs.Destroy();
  const auto desc_cleanup = aclmdlDestroyDesc(desc);
  result->resource_cleanup_status = input_cleanup != ACL_SUCCESS
                                        ? input_cleanup
                                        : output_cleanup != ACL_SUCCESS
                                              ? output_cleanup
                                              : desc_cleanup;
}

void WriteSummary(const std::string &path, const ProbeResult &result) {
  std::ofstream stream(path);
  stream << "{\n"
         << "  \"gate\": \"P3-MINI-FIA-OM-EXECUTE\",\n"
         << "  \"pass\": " << (result.pass ? "true" : "false") << ",\n"
         << "  \"build_initialize_status\": "
         << result.build_initialize_status << ",\n"
         << "  \"air_load_status\": " << result.air_load_status << ",\n"
         << "  \"build_status\": " << result.build_status << ",\n"
         << "  \"save_status\": " << result.save_status << ",\n"
         << "  \"model_buffer_bytes\": " << result.model_buffer_bytes
         << ",\n"
         << "  \"om_bytes\": " << result.om_bytes << ",\n"
         << "  \"acl_init_status\": " << result.acl_init_status << ",\n"
         << "  \"set_device_status\": " << result.set_device_status << ",\n"
         << "  \"load_status\": " << result.load_status << ",\n"
         << "  \"desc_status\": " << result.desc_status << ",\n"
         << "  \"input_count\": " << result.input_count << ",\n"
         << "  \"output_count\": " << result.output_count << ",\n"
         << "  \"io_contract_exact\": "
         << (result.io_contract_exact ? "true" : "false") << ",\n"
         << "  \"dataset_status\": " << result.dataset_status << ",\n"
         << "  \"execute_status\": " << result.execute_status << ",\n"
         << "  \"output_copy_status\": " << result.output_copy_status
         << ",\n"
         << "  \"output_exact\": "
         << (result.output_exact ? "true" : "false") << ",\n"
         << "  \"resource_cleanup_status\": "
         << result.resource_cleanup_status << ",\n"
         << "  \"unload_status\": " << result.unload_status << ",\n"
         << "  \"reset_device_status\": " << result.reset_device_status
         << ",\n"
         << "  \"acl_finalize_status\": " << result.acl_finalize_status
         << ",\n"
         << "  \"claim_boundary\": \"Weight-free FIA AIR-to-OM build plus one exact ACL execution; no GraphPp or P3 Owner claim.\"\n"
         << "}\n";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: mini_fia_om_probe AIR_PATH OM_PREFIX SUMMARY_JSON"
              << std::endl;
    return 2;
  }
  const std::string air_path = argv[1];
  const std::string om_prefix = argv[2];
  const std::string om_path = om_prefix + ".om";
  const std::string summary_path = argv[3];
  if (access(air_path.c_str(), R_OK) != 0 || access(om_path.c_str(), F_OK) == 0) {
    return 2;
  }

  ProbeResult result;
  std::vector<uint8_t> om;
  std::map<ge::AscendString, ge::AscendString> global_options = {
      {"ge.socVersion", "Ascend910B2"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
  };
  const auto initialize = ge::aclgrphBuildInitialize(global_options);
  result.build_initialize_status = StatusValue(initialize);
  if (initialize == ge::GRAPH_SUCCESS) {
    ge::Graph graph("MiniFIAOfflineModel");
    const auto air_load = graph.LoadFromFile(air_path.c_str());
    result.air_load_status = StatusValue(air_load);
    if (air_load == ge::GRAPH_SUCCESS && graph.IsValid()) {
      std::map<ge::AscendString, ge::AscendString> build_options = {
          {"ge.exec.precision_mode", "must_keep_origin_dtype"},
      };
      ge::ModelBufferData model;
      const auto build = ge::aclgrphBuildModel(graph, build_options, model);
      result.build_status = StatusValue(build);
      result.model_buffer_bytes = model.length;
      if (build == ge::GRAPH_SUCCESS && model.data != nullptr &&
          model.length > 0) {
        const auto save = ge::aclgrphSaveModel(om_prefix.c_str(), model);
        result.save_status = StatusValue(save);
        if (save == ge::GRAPH_SUCCESS) om = ReadBinary(om_path);
      }
    }
    ge::aclgrphBuildFinalize();
  }
  result.om_bytes = om.size();

  if (!om.empty()) {
    result.acl_init_status = aclInit(nullptr);
    if (result.acl_init_status == ACL_SUCCESS) {
      result.set_device_status = aclrtSetDevice(0);
      if (result.set_device_status == ACL_SUCCESS) {
        uint32_t model_id = 0;
        result.load_status =
            aclmdlLoadFromMem(om.data(), om.size(), &model_id);
        if (result.load_status == ACL_SUCCESS) {
          ExecuteModel(model_id, &result);
          result.unload_status = aclmdlUnload(model_id);
        }
        result.reset_device_status = aclrtResetDevice(0);
      }
      result.acl_finalize_status = aclFinalize();
    }
  }

  result.pass =
      result.build_initialize_status == ge::GRAPH_SUCCESS &&
      result.air_load_status == ge::GRAPH_SUCCESS &&
      result.build_status == ge::GRAPH_SUCCESS &&
      result.save_status == ge::GRAPH_SUCCESS &&
      result.model_buffer_bytes > 0 && result.om_bytes > 0 &&
      result.acl_init_status == ACL_SUCCESS &&
      result.set_device_status == ACL_SUCCESS &&
      result.load_status == ACL_SUCCESS && result.desc_status == ACL_SUCCESS &&
      result.io_contract_exact && result.dataset_status == ACL_SUCCESS &&
      result.execute_status == ACL_SUCCESS &&
      result.output_copy_status == ACL_SUCCESS && result.output_exact &&
      result.resource_cleanup_status == ACL_SUCCESS &&
      result.unload_status == ACL_SUCCESS &&
      result.reset_device_status == ACL_SUCCESS &&
      result.acl_finalize_status == ACL_SUCCESS;
  WriteSummary(summary_path, result);
  std::cout << "MINI_FIA_OM_RESULT pass=" << result.pass
            << " build_status=" << result.build_status
            << " load_status=" << result.load_status
            << " execute_status=" << result.execute_status
            << " output_exact=" << result.output_exact
            << " om_bytes=" << result.om_bytes << std::endl;
  return result.pass ? 0 : 10;
}
