#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "graph/graph.h"

namespace {
constexpr int32_t kFeedTimeoutMs = 600000;
constexpr int32_t kFetchTimeoutMs = 600000;
constexpr size_t kCacheBytes =
    28ULL * 2ULL * 128ULL * 4ULL * 128ULL * sizeof(uint16_t);
constexpr size_t kLogitsBytes = 152064ULL * sizeof(float);

struct InputSpec {
  const char *name;
  std::vector<int64_t> shape;
  ge::DataType dtype;
  size_t bytes;
};

const std::array<InputSpec, 8> kInputSpecs = {{
    {"token_id.bin", {1, 1}, ge::DT_INT64, sizeof(int64_t)},
    {"position.bin", {1}, ge::DT_INT64, sizeof(int64_t)},
    {"sequence_length.bin", {1, 1}, ge::DT_INT32, sizeof(int32_t)},
    {"key_cache.bin", {28, 2, 128, 4, 128}, ge::DT_BF16, kCacheBytes},
    {"slot_mapping.bin", {1}, ge::DT_INT32, sizeof(int32_t)},
    {"block_table.bin", {1, 2}, ge::DT_INT32, 2 * sizeof(int32_t)},
    {"value_cache.bin", {28, 2, 128, 4, 128}, ge::DT_BF16, kCacheBytes},
    {"explicit_tiling.bin", {72}, ge::DT_UINT8, 72},
}};

const std::array<const char *, 4> kOutputNames = {{
    "logits.bin", "key_cache.bin", "value_cache.bin", "next_position.bin"}};
const std::array<size_t, 4> kOutputBytes = {{
    kLogitsBytes, kCacheBytes, kCacheBytes, sizeof(int64_t)}};
const std::array<ge::DataType, 4> kOutputDtypes = {{
    ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_INT64}};

bool ReadFile(const std::string &path, size_t expected,
              std::vector<uint8_t> &data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const auto size = static_cast<size_t>(stream.tellg());
  if (size != expected) {
    std::cerr << "INPUT_SIZE_MISMATCH path=" << path << " actual=" << size
              << " expected=" << expected << std::endl;
    return false;
  }
  data.resize(size);
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(data.data()),
              static_cast<std::streamsize>(size));
  return static_cast<bool>(stream);
}

ge::Tensor MakeTensor(const InputSpec &spec, std::vector<uint8_t> &data) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(
      ge::TensorDesc(ge::Shape(spec.shape), ge::FORMAT_ND, spec.dtype));
  tensor.SetData(data.data(), data.size());
  return tensor;
}

bool WriteTensor(const std::string &path, const ge::Tensor &tensor,
                 size_t expected_bytes, ge::DataType expected_dtype) {
  const auto actual_dtype = tensor.GetTensorDesc().GetDataType();
  if (tensor.GetData() == nullptr || tensor.GetSize() != expected_bytes ||
      actual_dtype != expected_dtype) {
    std::cerr << "OUTPUT_CONTRACT_MISMATCH path=" << path
              << " size=" << tensor.GetSize()
              << " expected_size=" << expected_bytes
              << " dtype=" << static_cast<int>(actual_dtype)
              << " expected_dtype=" << static_cast<int>(expected_dtype)
              << std::endl;
    return false;
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return false;
  stream.write(reinterpret_cast<const char *>(tensor.GetData()),
               static_cast<std::streamsize>(tensor.GetSize()));
  return static_cast<bool>(stream);
}

ge::dflow::FlowGraph BuildDataFlow(const std::string &air_path,
                                   const std::string &compile_config) {
  ge::dflow::FlowGraph flow_graph("attempt66a_r4_dataflow_smoke");
  auto data0 = ge::dflow::FlowData("input0", 0);
  auto data1 = ge::dflow::FlowData("input1", 1);
  auto data2 = ge::dflow::FlowData("input2", 2);
  auto data3 = ge::dflow::FlowData("input3", 3);
  auto data4 = ge::dflow::FlowData("input4", 4);
  auto data5 = ge::dflow::FlowData("input5", 5);
  auto data6 = ge::dflow::FlowData("input6", 6);
  auto data7 = ge::dflow::FlowData("input7", 7);

  auto graph_pp = ge::dflow::GraphPp(
      "attempt65_full_decoder_air", [air_path]() {
        ge::Graph graph("Attempt65FullDecoderForDataFlow");
        const auto status = graph.LoadFromFile(air_path.c_str());
        std::cout << "AIR_LOAD status=" << status
                  << " valid=" << graph.IsValid() << std::endl;
        return graph;
      });
  graph_pp.SetCompileConfig(compile_config.c_str());
  auto node = ge::dflow::FlowNode("attempt65_decoder_node", 8, 4);
  node.AddPp(graph_pp)
      .SetInput(0, data0)
      .SetInput(1, data1)
      .SetInput(2, data2)
      .SetInput(3, data3)
      .SetInput(4, data4)
      .SetInput(5, data5)
      .SetInput(6, data6)
      .SetInput(7, data7);

  std::vector<ge::dflow::FlowOperator> inputs = {
      data0, data1, data2, data3, data4, data5, data6, data7};
  std::vector<std::pair<ge::dflow::FlowOperator, std::vector<size_t>>> outputs = {
      {node, {0, 1, 2, 3}}};
  flow_graph.SetInputs(inputs).SetOutputs(outputs);
  return flow_graph;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: dataflow_full_decoder_smoke AIR GRAPH_CONFIG "
                 "INPUT_DIR OUTPUT_DIR\n";
    return 2;
  }
  const std::string air_path = argv[1];
  const std::string graph_config = argv[2];
  const std::string input_dir = argv[3];
  const std::string output_dir = argv[4];
  auto flow_graph = BuildDataFlow(air_path, graph_config);

  std::map<ge::AscendString, ge::AscendString> options = {
      {"ge.exec.deviceId", "0"},
      {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
      {"ge.exec.logicalDeviceId", "[0:0]"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"},
      {"ge.graphRunMode", "0"}};
  auto ret = ge::GEInitialize(options);
  if (ret != ge::SUCCESS) {
    std::cerr << "GE_INITIALIZE_FAILED ret=" << ret << std::endl;
    return 3;
  }
  auto session = std::make_shared<ge::Session>(
      std::map<ge::AscendString, ge::AscendString>{});
  ret = session->AddGraph(0, flow_graph.ToGeGraph());
  if (ret != ge::SUCCESS) {
    std::cerr << "ADD_GRAPH_FAILED ret=" << ret << std::endl;
    session.reset();
    ge::GEFinalize();
    return 4;
  }

  std::array<std::vector<uint8_t>, 8> buffers;
  std::vector<ge::Tensor> inputs;
  inputs.reserve(kInputSpecs.size());
  for (size_t index = 0; index < kInputSpecs.size(); ++index) {
    const auto &spec = kInputSpecs[index];
    if (!ReadFile(input_dir + "/" + spec.name, spec.bytes, buffers[index])) {
      session.reset();
      ge::GEFinalize();
      return 5;
    }
    inputs.push_back(MakeTensor(spec, buffers[index]));
    std::cout << "INPUT index=" << index
              << " dtype=" << static_cast<int>(spec.dtype)
              << " bytes=" << spec.bytes << std::endl;
  }

  ge::DataFlowInfo flow_info;
  ret = session->FeedDataFlowGraph(0, inputs, flow_info, kFeedTimeoutMs);
  if (ret != ge::SUCCESS) {
    std::cerr << "FEED_FAILED ret=" << ret << std::endl;
    session.reset();
    ge::GEFinalize();
    return 6;
  }
  std::vector<ge::Tensor> outputs;
  ret = session->FetchDataFlowGraph(0, outputs, flow_info, kFetchTimeoutMs);
  if (ret != ge::SUCCESS || outputs.size() != kOutputNames.size()) {
    std::cerr << "FETCH_FAILED ret=" << ret
              << " outputs=" << outputs.size() << std::endl;
    session.reset();
    ge::GEFinalize();
    return 7;
  }

  std::ofstream metadata(output_dir + "/runtime-metadata.tsv",
                         std::ios::trunc);
  metadata << "index\tname\tdtype\tbytes\n";
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (!WriteTensor(output_dir + "/" + kOutputNames[index], outputs[index],
                     kOutputBytes[index], kOutputDtypes[index])) {
      session.reset();
      ge::GEFinalize();
      return 8;
    }
    metadata << index << '\t' << kOutputNames[index] << '\t'
             << static_cast<int>(outputs[index].GetTensorDesc().GetDataType())
             << '\t' << outputs[index].GetSize() << '\n';
  }
  metadata.close();
  session.reset();
  ge::GEFinalize();
  std::cout << "ATTEMPT66A_R4_CPP_DATAFLOW_COMPLETE feed_calls=1 fetch_calls=1"
            << std::endl;
  return 0;
}
