#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "graph/graph.h"

namespace {
constexpr uint32_t kGraphId = 0;
constexpr int32_t kFeedTimeoutMs = 120000;
constexpr int32_t kFetchTimeoutMs = 120000;
constexpr size_t kBatchSize = 4U;
constexpr size_t kKvHeads = 4U;
constexpr size_t kHeadDim = 128U;
constexpr size_t kBlockSize = 128U;
constexpr size_t kBlocksPerRow = 3U;
constexpr size_t kPhysicalBlocks = kBatchSize * kBlocksPerRow;
constexpr size_t kPackWidth = 16U;
constexpr size_t kPackedHeads = kKvHeads * kHeadDim / kPackWidth;
constexpr size_t kRowElements = kKvHeads * kHeadDim;
constexpr size_t kCacheElements =
    kPhysicalBlocks * kPackedHeads * kBlockSize * kPackWidth;
constexpr size_t kCacheBytes = kCacheElements * sizeof(uint16_t);
const uint16_t kKeyBits[kBatchSize] = {0x3f80, 0x4000, 0x4040, 0x4080};
const uint16_t kValueBits[kBatchSize] = {0xbf80, 0xc000, 0xc040, 0xc080};

struct InputBuffers {
  std::vector<uint16_t> key = std::vector<uint16_t>(kBatchSize * kRowElements);
  std::vector<uint16_t> value =
      std::vector<uint16_t>(kBatchSize * kRowElements);
  std::vector<uint16_t> key_cache = std::vector<uint16_t>(kCacheElements, 0);
  std::vector<uint16_t> value_cache =
      std::vector<uint16_t>(kCacheElements, 0);
  std::vector<int32_t> slots = {0, 384, 768, 1152};

  InputBuffers() {
    for (size_t row = 0; row < kBatchSize; ++row) {
      std::fill(key.begin() + row * kRowElements,
                key.begin() + (row + 1) * kRowElements, kKeyBits[row]);
      std::fill(value.begin() + row * kRowElements,
                value.begin() + (row + 1) * kRowElements, kValueBits[row]);
    }
  }
};

ge::Tensor MakeTensor(const std::vector<int64_t> &shape, ge::DataType dtype,
                      const void *data, size_t bytes) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype));
  tensor.SetData(reinterpret_cast<const uint8_t *>(data), bytes);
  return tensor;
}

std::vector<ge::Tensor> MakeInputs(InputBuffers &buffers) {
  return {
      MakeTensor({4, 4, 128}, ge::DT_BF16, buffers.key.data(),
                 buffers.key.size() * sizeof(uint16_t)),
      MakeTensor({4, 4, 128}, ge::DT_BF16, buffers.value.data(),
                 buffers.value.size() * sizeof(uint16_t)),
      MakeTensor({12, 32, 128, 16}, ge::DT_BF16, buffers.key_cache.data(),
                 buffers.key_cache.size() * sizeof(uint16_t)),
      MakeTensor({12, 32, 128, 16}, ge::DT_BF16,
                 buffers.value_cache.data(),
                 buffers.value_cache.size() * sizeof(uint16_t)),
      MakeTensor({4}, ge::DT_INT32, buffers.slots.data(),
                 buffers.slots.size() * sizeof(int32_t)),
  };
}

ge::Graph LoadAir(const std::string &air_path, uint32_t *load_status = nullptr,
                  bool *valid = nullptr) {
  ge::Graph graph("CruiseKvAliasProbe");
  const auto status = graph.LoadFromFile(air_path.c_str());
  if (load_status != nullptr) *load_status = static_cast<uint32_t>(status);
  if (valid != nullptr) *valid = graph.IsValid();
  std::cout << "KV_ALIAS_AIR_LOAD status=" << status
            << " valid=" << graph.IsValid() << std::endl;
  return graph;
}

ge::dflow::FlowGraph BuildFlowGraph(const std::string &air_path,
                                    const std::string &graph_config) {
  using namespace ge::dflow;
  auto key = FlowData("key", 0);
  auto value = FlowData("value", 1);
  auto key_cache = FlowData("key_cache", 2);
  auto value_cache = FlowData("value_cache", 3);
  auto slots = FlowData("slots", 4);
  auto update = GraphPp("kv_alias_graph_pp",
                        [air_path]() { return LoadAir(air_path); })
                    .SetCompileConfig(graph_config.c_str());
  auto node = FlowNode("kv_alias_node", 5, 2);
  node.AddPp(update)
      .SetInput(0, key)
      .SetInput(1, value)
      .SetInput(2, key_cache)
      .SetInput(3, value_cache)
      .SetInput(4, slots);
  FlowGraph graph("cruise_kv_alias_dataflow");
  graph.SetInputs({key, value, key_cache, value_cache, slots})
      .SetOutputs({{node, {0, 1}}})
      .SetContainsNMappingNode(true);
  return graph;
}

std::vector<uint16_t> ExpectedCache(const uint16_t bits[kBatchSize]) {
  std::vector<uint16_t> expected(kCacheElements, 0);
  for (size_t row = 0; row < kBatchSize; ++row) {
    const size_t block = row * kBlocksPerRow;
    for (size_t packed_head = 0; packed_head < kPackedHeads; ++packed_head) {
      for (size_t lane = 0; lane < kPackWidth; ++lane) {
        const size_t index =
            (((block * kPackedHeads + packed_head) * kBlockSize) * kPackWidth) +
            lane;
        expected[index] = bits[row];
      }
    }
  }
  return expected;
}

bool TensorExact(const ge::Tensor &output, const std::vector<uint16_t> &expected) {
  if (output.GetData() == nullptr || output.GetSize() != kCacheBytes) return false;
  const auto *actual = reinterpret_cast<const uint16_t *>(output.GetData());
  return std::equal(expected.begin(), expected.end(), actual);
}

bool WriteOutputs(const std::string &path,
                  const std::vector<ge::Tensor> &outputs) {
  if (outputs.size() != 2 || outputs[0].GetData() == nullptr ||
      outputs[1].GetData() == nullptr || outputs[0].GetSize() != kCacheBytes ||
      outputs[1].GetSize() != kCacheBytes) {
    return false;
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return false;
  for (const auto &output : outputs) {
    stream.write(reinterpret_cast<const char *>(output.GetData()),
                 static_cast<std::streamsize>(output.GetSize()));
  }
  return static_cast<bool>(stream);
}

void WriteSummary(const std::string &path, const std::string &mode,
                  ge::Status status, ge::Status add_status,
                  ge::Status compile_status, ge::Status feed_status,
                  ge::Status fetch_status, uint32_t model_load_status,
                  bool model_valid, size_t output_count, bool key_exact,
                  bool value_exact, bool output_saved, int64_t elapsed_ms) {
  const bool pass = status == ge::SUCCESS &&
                    model_load_status == ge::GRAPH_SUCCESS && model_valid &&
                    output_count == 2 && key_exact && value_exact && output_saved;
  std::ofstream stream(path, std::ios::trunc);
  stream << "{\n"
         << "  \"gate\": \"V2-KV-ALIAS-GRAPHPP\",\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"mode\": \"" << mode << "\",\n"
         << "  \"status\": " << static_cast<uint32_t>(status) << ",\n"
         << "  \"add_status\": " << static_cast<uint32_t>(add_status)
         << ",\n"
         << "  \"compile_status\": "
         << static_cast<uint32_t>(compile_status) << ",\n"
         << "  \"feed_status\": " << static_cast<uint32_t>(feed_status)
         << ",\n"
         << "  \"fetch_status\": " << static_cast<uint32_t>(fetch_status)
         << ",\n"
         << "  \"model_load_status\": " << model_load_status << ",\n"
         << "  \"model_valid\": " << (model_valid ? "true" : "false")
         << ",\n"
         << "  \"output_count\": " << output_count << ",\n"
         << "  \"key_exact\": " << (key_exact ? "true" : "false")
         << ",\n"
         << "  \"value_exact\": " << (value_exact ? "true" : "false")
         << ",\n"
         << "  \"output_exact\": "
         << (key_exact && value_exact ? "true" : "false") << ",\n"
         << "  \"output_saved\": " << (output_saved ? "true" : "false")
         << ",\n"
         << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
         << "  \"claim_boundary\": \"Single-layer B4/K384 KV update execution only; no full-model, shared-lease, Device-copy, or P5 performance claim.\"\n"
         << "}\n";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: kv_alias_graphpp_probe MODE AIR GRAPH_CONFIG "
                 "DEPLOY_CONFIG OUTPUT SUMMARY_JSON"
              << std::endl;
    return 2;
  }
  const std::string mode = argv[1];
  const std::string air_path = argv[2];
  const std::string graph_config = argv[3];
  const std::string deploy_config = argv[4];
  const std::string output_path = argv[5];
  const std::string summary_path = argv[6];
  if ((mode != "graph" && mode != "dataflow") ||
      access(air_path.c_str(), R_OK) != 0 ||
      access(graph_config.c_str(), R_OK) != 0 ||
      access(deploy_config.c_str(), R_OK) != 0) {
    return 2;
  }

  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
  };
  if (mode == "dataflow") {
    config["ge.exec.logicalDeviceClusterDeployMode"] = "SINGLE";
    config["ge.exec.logicalDeviceId"] = "[0:0]";
    config["ge.experiment.data_flow_deploy_info_path"] = deploy_config.c_str();
  }

  auto status = ge::GEInitialize(config);
  ge::Status add_status = ge::FAILED;
  ge::Status compile_status = ge::FAILED;
  ge::Status feed_status = ge::FAILED;
  ge::Status fetch_status = ge::FAILED;
  uint32_t model_load_status = UINT32_MAX;
  bool model_valid = false;
  if (status != ge::SUCCESS) {
    WriteSummary(summary_path, mode, status, add_status, compile_status,
                 feed_status, fetch_status, model_load_status, model_valid, 0,
                 false, false, false, 0);
    return 10;
  }

  auto session = std::make_shared<ge::Session>(config);
  const auto started = std::chrono::steady_clock::now();
  auto source_graph = LoadAir(air_path, &model_load_status, &model_valid);
  InputBuffers buffers;
  auto inputs = MakeInputs(buffers);
  std::vector<ge::Tensor> outputs;
  if (mode == "graph") {
    add_status = session->AddGraph(kGraphId, source_graph);
    status = add_status;
    if (status == ge::SUCCESS) {
      status = session->RunGraph(kGraphId, inputs, outputs);
    }
  } else {
    auto flow_graph = BuildFlowGraph(air_path, graph_config);
    add_status = session->AddGraph(kGraphId, flow_graph.ToGeGraph());
    status = add_status;
    if (status == ge::SUCCESS) {
      compile_status = session->CompileGraph(kGraphId);
      status = compile_status;
    }
    if (status == ge::SUCCESS) {
      ge::DataFlowInfo flow_info;
      feed_status = session->FeedDataFlowGraph(kGraphId, inputs, flow_info,
                                               kFeedTimeoutMs);
      status = feed_status;
      if (status == ge::SUCCESS) {
        fetch_status = session->FetchDataFlowGraph(kGraphId, outputs, flow_info,
                                                   kFetchTimeoutMs);
        status = fetch_status;
      }
    }
  }
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - started)
                              .count();
  const auto expected_key = ExpectedCache(kKeyBits);
  const auto expected_value = ExpectedCache(kValueBits);
  const bool key_exact =
      status == ge::SUCCESS && outputs.size() == 2 &&
      TensorExact(outputs[0], expected_key);
  const bool value_exact =
      status == ge::SUCCESS && outputs.size() == 2 &&
      TensorExact(outputs[1], expected_value);
  const bool output_saved =
      key_exact && value_exact && WriteOutputs(output_path, outputs);
  WriteSummary(summary_path, mode, status, add_status, compile_status,
               feed_status, fetch_status, model_load_status, model_valid,
               outputs.size(), key_exact, value_exact, output_saved, elapsed_ms);
  std::cout << "KV_ALIAS_GRAPHPP_RESULT mode=" << mode
            << " status=" << static_cast<uint32_t>(status)
            << " key_exact=" << key_exact
            << " value_exact=" << value_exact
            << " output_saved=" << output_saved
            << " elapsed_ms=" << elapsed_ms << std::endl;
  if (add_status == ge::SUCCESS) session->RemoveGraph(kGraphId);
  session.reset();
  ge::GEFinalize();
  return status == ge::SUCCESS && model_load_status == ge::GRAPH_SUCCESS &&
                 model_valid && key_exact && value_exact && output_saved
             ? 0
             : 10;
}
