#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "ge/ge_api.h"
#include "graph/graph.h"

namespace {
constexpr size_t kRows = 4;
constexpr size_t kLayers = 28;
constexpr size_t kBlocksPerRequest = 3;
constexpr size_t kPhysicalBlocks = kRows * kBlocksPerRequest;
constexpr size_t kBlockSize = 128;
constexpr size_t kKvHeads = 4;
constexpr size_t kHeadDim = 128;
constexpr size_t kCacheElements =
    kLayers * kPhysicalBlocks * kBlockSize * kKvHeads * kHeadDim;
constexpr size_t kCacheBytes = kCacheElements * sizeof(uint16_t);
constexpr size_t kBlockElements = kBlockSize * kKvHeads * kHeadDim;
constexpr int64_t kDefaultEos = 151645;

struct Row {
  bool active = false;
  bool ignore_eos = false;
  int64_t target_steps = 0;
  int64_t eos_token = kDefaultEos;
  int64_t prompt_index = 0;
  int64_t commits = 0;
  int64_t token = 0;
  int64_t position = 0;
  int64_t finish_reason = 0;
  std::vector<int64_t> prompt;
  std::vector<int64_t> tokens;
};

ge::Tensor MakeTensor(std::vector<uint8_t> &data,
                      const std::vector<int64_t> &shape,
                      ge::DataType dtype) {
  ge::Tensor tensor;
  tensor.SetTensorDesc(ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype));
  tensor.SetData(data.data(), data.size());
  return tensor;
}

template <typename T>
std::vector<uint8_t> Bytes(const std::vector<T> &values) {
  std::vector<uint8_t> data(values.size() * sizeof(T));
  std::memcpy(data.data(), values.data(), data.size());
  return data;
}

bool CopyOutput(const ge::Tensor &tensor, size_t bytes, ge::DataType dtype,
                std::vector<uint8_t> &destination) {
  if (tensor.GetData() == nullptr || tensor.GetSize() != bytes ||
      tensor.GetTensorDesc().GetDataType() != dtype) {
    return false;
  }
  const auto *begin = static_cast<const uint8_t *>(tensor.GetData());
  destination.assign(begin, begin + bytes);
  return true;
}

std::vector<std::vector<int64_t>> MakePrompts(const std::string &scenario) {
  if (scenario == "b1-short") {
    return {{9707, 9708, 9709, 9710}};
  }
  if (scenario == "b1-stock") {
    return {{1000, 1001, 1002, 1003}};
  }
  if (scenario == "b1-eos") {
    return {{151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553,
             54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645,
             198, 151644, 872, 198, 5598, 458, 4287, 2033, 13, 3155,
             537, 2550, 894, 1467, 13, 151645, 198, 151644, 77091, 198}};
  }
  if (scenario == "b1-full") {
    std::vector<int64_t> prompt(128);
    for (int64_t index = 0; index < 128; ++index) prompt[index] = 1000 + index;
    return {prompt};
  }
  if (scenario == "b4-mixed") {
    std::vector<std::vector<int64_t>> prompts;
    for (int64_t length : {1LL, 7LL, 32LL, 128LL}) {
      std::vector<int64_t> prompt(static_cast<size_t>(length));
      for (int64_t index = 0; index < length; ++index) {
        prompt[static_cast<size_t>(index)] = 2000 + length + index;
      }
      prompts.push_back(std::move(prompt));
    }
    return prompts;
  }
  return {};
}

std::array<int64_t, kRows> Targets(const std::string &scenario) {
  if (scenario == "b1-short" || scenario == "b1-stock" ||
      scenario == "b1-eos") {
    return {8, 0, 0, 0};
  }
  if (scenario == "b1-full") return {256, 0, 0, 0};
  return {8, 16, 64, 256};
}

int32_t Slot(size_t row, int64_t position) {
  const int32_t physical = static_cast<int32_t>(
      row * kBlocksPerRequest + static_cast<size_t>(position / kBlockSize));
  return physical * static_cast<int32_t>(kBlockSize) +
         static_cast<int32_t>(position % kBlockSize);
}

void Adler32Update(const uint8_t *data, size_t bytes, uint32_t &first,
                   uint32_t &second) {
  constexpr uint32_t kModulus = 65521;
  while (bytes > 0) {
    const size_t chunk = bytes > 5552 ? 5552 : bytes;
    for (size_t index = 0; index < chunk; ++index) {
      first += data[index];
      second += first;
    }
    first %= kModulus;
    second %= kModulus;
    data += chunk;
    bytes -= chunk;
  }
}

uint32_t RowChecksum(const std::vector<uint8_t> &key,
                     const std::vector<uint8_t> &value, size_t row) {
  uint32_t first = 1;
  uint32_t second = 0;
  for (const auto *cache : {&key, &value}) {
    for (size_t layer = 0; layer < kLayers; ++layer) {
      for (size_t page = 0; page < kBlocksPerRequest; ++page) {
        const size_t physical = row * kBlocksPerRequest + page;
        const size_t offset =
            (layer * kPhysicalBlocks + physical) * kBlockElements * sizeof(uint16_t);
        Adler32Update(cache->data() + offset,
                      kBlockElements * sizeof(uint16_t), first, second);
      }
    }
  }
  return (second << 16) | first;
}

bool HasActiveRow(const std::array<Row, kRows> &rows) {
  for (const auto &row : rows) {
    if (row.active) return true;
  }
  return false;
}

bool RunStep(const std::shared_ptr<ge::Session> &session,
             std::array<Row, kRows> &rows, std::vector<uint8_t> &key_cache,
             std::vector<uint8_t> &value_cache, int64_t &model_calls) {
  std::vector<int64_t> token(kRows);
  std::vector<int64_t> position(kRows);
  std::vector<int32_t> length(kRows);
  std::vector<int32_t> slot(kRows);
  std::vector<int32_t> active(kRows);
  std::vector<int32_t> block_table(kRows * kBlocksPerRequest);
  for (size_t row_index = 0; row_index < kRows; ++row_index) {
    const auto &row = rows[row_index];
    const bool ingesting = row.prompt_index < static_cast<int64_t>(row.prompt.size());
    const int64_t current_position = ingesting ? row.prompt_index : row.position;
    token[row_index] = ingesting
                           ? row.prompt[static_cast<size_t>(row.prompt_index)]
                           : row.token;
    position[row_index] = current_position;
    length[row_index] = static_cast<int32_t>(current_position + 1);
    slot[row_index] = Slot(row_index, current_position);
    active[row_index] = row.active ? 1 : 0;
    for (size_t page = 0; page < kBlocksPerRequest; ++page) {
      block_table[row_index * kBlocksPerRequest + page] =
          static_cast<int32_t>(row_index * kBlocksPerRequest + page);
    }
  }

  auto token_bytes = Bytes(token);
  auto position_bytes = Bytes(position);
  auto length_bytes = Bytes(length);
  auto slot_bytes = Bytes(slot);
  auto active_bytes = Bytes(active);
  auto block_bytes = Bytes(block_table);
  // RunGraph binds this imported AIR by its audited Data-node indexes.
  std::vector<ge::Tensor> inputs = {
      MakeTensor(token_bytes, {4, 1}, ge::DT_INT64),
      MakeTensor(position_bytes, {4}, ge::DT_INT64),
      MakeTensor(length_bytes, {4, 1}, ge::DT_INT32),
      MakeTensor(key_cache, {28, 12, 128, 4, 128}, ge::DT_BF16),
      MakeTensor(slot_bytes, {4}, ge::DT_INT32),
      MakeTensor(active_bytes, {4}, ge::DT_INT32),
      MakeTensor(block_bytes, {4, 3}, ge::DT_INT32),
      MakeTensor(value_cache, {28, 12, 128, 4, 128}, ge::DT_BF16)};
  std::vector<ge::Tensor> outputs;
  const auto status = session->RunGraph(0, inputs, outputs);
  ++model_calls;
  if (status != ge::SUCCESS || outputs.size() != 4 ||
      outputs[0].GetData() == nullptr || outputs[0].GetSize() != kRows * sizeof(int64_t) ||
      outputs[0].GetTensorDesc().GetDataType() != ge::DT_INT64 ||
      outputs[3].GetData() == nullptr || outputs[3].GetSize() != kRows * sizeof(int64_t) ||
      outputs[3].GetTensorDesc().GetDataType() != ge::DT_INT64 ||
      !CopyOutput(outputs[1], kCacheBytes, ge::DT_BF16, key_cache) ||
      !CopyOutput(outputs[2], kCacheBytes, ge::DT_BF16, value_cache)) {
    std::cerr << "P3_ORACLE_RUN_FAILED status=" << status
              << " outputs=" << outputs.size() << std::endl;
    return false;
  }
  const auto *generated = reinterpret_cast<const int64_t *>(outputs[0].GetData());
  const auto *next_position =
      reinterpret_cast<const int64_t *>(outputs[3].GetData());
  for (size_t row_index = 0; row_index < kRows; ++row_index) {
    auto &row = rows[row_index];
    if (!row.active) continue;
    if (next_position[row_index] != row.position + 1) return false;
    row.position = next_position[row_index];
    row.token = generated[row_index];
    const bool prompt_step =
        row.prompt_index < static_cast<int64_t>(row.prompt.size());
    if (prompt_step) ++row.prompt_index;
    const bool commit = !prompt_step ||
                        row.prompt_index == static_cast<int64_t>(row.prompt.size());
    if (!commit) continue;
    row.tokens.push_back(generated[row_index]);
    ++row.commits;
    const bool eos = generated[row_index] == row.eos_token && !row.ignore_eos;
    if (eos || row.commits == row.target_steps) {
      row.active = false;
      row.finish_reason = eos ? 1 : 2;
    }
  }
  return true;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: p3_graph_oracle AIR SCENARIO SUMMARY_JSON" << std::endl;
    return 2;
  }
  const std::string air_path = argv[1];
  const std::string scenario = argv[2];
  const std::string summary_path = argv[3];
  const char *external_weight_env = std::getenv("CRUISE_P3_EXTERNAL_WEIGHT_DIR");
  const std::string external_weights =
      external_weight_env == nullptr ? std::string() : external_weight_env;
  const char *asset_root_env = std::getenv("CRUISE_P3_ASSET_ROOT");
  const std::string asset_root =
      asset_root_env == nullptr ? "/workspace/cruise-assets" : asset_root_env;
  const char *run_root_env = std::getenv("CRUISE_P3_RUN_ROOT");
  const std::string run_root =
      run_root_env == nullptr ? std::string() : run_root_env;
  const bool controlled_runtime_view =
      !run_root.empty() && run_root.back() != '/' &&
      external_weights == run_root + "/.ge-external-view";
  const bool external_weights_allowed =
      external_weights.rfind("/dev/shm/", 0) == 0 ||
      (!asset_root.empty() && asset_root.back() != '/' &&
       external_weights.rfind(asset_root + "/", 0) == 0) ||
      controlled_runtime_view;
  const auto prompts = MakePrompts(scenario);
  if (access(air_path.c_str(), R_OK) != 0 || prompts.empty() ||
      !external_weights_allowed ||
      access(external_weights.c_str(), R_OK) != 0) {
    std::cerr << "invalid oracle AIR, scenario, or external weights" << std::endl;
    return 2;
  }

  ge::Graph graph("P3ColdGraphOracle");
  const auto load_status = graph.LoadFromFile(air_path.c_str());
  std::cout << "P3_ORACLE_LOAD status=" << load_status
            << " valid=" << graph.IsValid() << std::endl;
  if (load_status != ge::GRAPH_SUCCESS || !graph.IsValid()) return 3;
  std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.externalWeight", "1"},
      {"ge.externalWeightDir", external_weights.c_str()},
      {"ge.modelFileNamePrefix", "p3_graph_oracle"},
      {"ge.graphRunMode", "0"},
      {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
  auto status = ge::GEInitialize(config);
  if (status != ge::SUCCESS) return static_cast<int>(status);
  auto session = std::make_shared<ge::Session>(config);
  status = session->AddGraph(0, graph);
  if (status != ge::SUCCESS) {
    session.reset();
    ge::GEFinalize();
    return static_cast<int>(status);
  }

  std::array<Row, kRows> rows{};
  const auto targets = Targets(scenario);
  for (size_t index = 0; index < prompts.size(); ++index) {
    rows[index].active = true;
    rows[index].prompt = prompts[index];
    rows[index].target_steps = targets[index];
    rows[index].ignore_eos = scenario == "b1-full";
    rows[index].eos_token = kDefaultEos;
  }
  std::vector<uint8_t> key_cache(kCacheBytes, 0);
  std::vector<uint8_t> value_cache(kCacheBytes, 0);
  int64_t model_calls = 0;
  bool passed = true;
  while (HasActiveRow(rows)) {
    if (!RunStep(session, rows, key_cache, value_cache, model_calls)) {
      passed = false;
      break;
    }
    if (model_calls > 383) {
      passed = false;
      break;
    }
  }
  if (scenario == "b1-eos") {
    const auto &row = rows[0];
    passed = passed && row.commits == 1 && row.finish_reason == 1 &&
             row.tokens.size() == 1 && row.tokens[0] == kDefaultEos;
  }
  if (scenario == "b1-stock") {
    const std::vector<int64_t> expected = {355, 11, 220, 17, 15, 16, 16, 11};
    passed = passed && rows[0].tokens == expected;
  }
  session->RemoveGraph(0);
  session.reset();
  ge::GEFinalize();

  std::ofstream summary(summary_path, std::ios::trunc);
  if (!summary) return 4;
  summary << "{\n  \"gate\": \"P3-GRAPH-ORACLE\",\n"
          << "  \"pass\": " << (passed ? "true" : "false") << ",\n"
          << "  \"scenario\": \"" << scenario << "\",\n"
          << "  \"model_calls\": " << model_calls << ",\n"
          << "  \"request_states\": [";
  for (size_t index = 0; index < prompts.size(); ++index) {
    if (index != 0) summary << ", ";
    const auto &row = rows[index];
    summary << "{\"request\": " << 10000 + static_cast<int64_t>(index)
            << ", \"generation\": " << 1 + static_cast<int64_t>(index)
            << ", \"row\": " << index
            << ", \"commits\": " << row.commits
            << ", \"retired\": " << (!row.active ? "true" : "false")
            << ", \"cancelled\": false"
            << ", \"final_position\": " << row.position
            << ", \"final_page\": " << row.position / 128
            << ", \"final_checksum\": "
            << RowChecksum(key_cache, value_cache, index)
            << ", \"finish_reason\": " << row.finish_reason
            << ", \"tokens\": [";
    for (size_t token_index = 0; token_index < row.tokens.size(); ++token_index) {
      if (token_index != 0) summary << ", ";
      summary << row.tokens[token_index];
    }
    summary << "]}";
  }
  summary << "],\n"
          << "  \"claim_boundary\": \"Independent cold ge::Session::RunGraph recurrence; no Owner or performance claim.\"\n}\n";
  std::cout << "P3_ORACLE_RESULT pass=" << (passed ? 1 : 0)
            << " scenario=" << scenario << " model_calls=" << model_calls
            << std::endl;
  return passed ? 0 : 20;
}
