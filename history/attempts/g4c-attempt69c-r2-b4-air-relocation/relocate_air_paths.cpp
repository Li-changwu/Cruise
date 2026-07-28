#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "graph/graph.h"

namespace {
constexpr size_t kExpectedFileConstants = 342;

std::string ToString(const ge::AscendString &value) {
  const auto *text = value.GetString();
  return text == nullptr ? std::string() : std::string(text);
}

bool StartsWith(const std::string &value, const std::string &prefix) {
  return value.size() >= prefix.size() &&
         value.compare(0, prefix.size(), prefix) == 0;
}

bool Readable(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  return static_cast<bool>(stream);
}

bool NodeSignature(const ge::Graph &graph, std::vector<std::string> &signature) {
  signature.clear();
  for (const auto &node : graph.GetAllNodes()) {
    ge::AscendString name;
    ge::AscendString type;
    if (node.GetName(name) != ge::GRAPH_SUCCESS ||
        node.GetType(type) != ge::GRAPH_SUCCESS) {
      return false;
    }
    signature.push_back(ToString(name) + "\n" + ToString(type));
  }
  std::sort(signature.begin(), signature.end());
  return true;
}

bool AuditPaths(const ge::Graph &graph, const std::string &required_prefix,
                const std::string &forbidden_prefix, size_t &count) {
  count = 0;
  ge::AscendString file_path_attr("file_path");
  for (const auto &node : graph.GetAllNodes()) {
    ge::AscendString type;
    if (node.GetType(type) != ge::GRAPH_SUCCESS) return false;
    if (ToString(type) != "FileConstant") continue;
    ge::AscendString path;
    if (node.GetAttr(file_path_attr, path) != ge::GRAPH_SUCCESS) return false;
    const std::string value = ToString(path);
    if (!StartsWith(value, required_prefix + "/") ||
        StartsWith(value, forbidden_prefix + "/") || !Readable(value)) {
      std::cerr << "PATH_AUDIT_FAILED path=" << value << std::endl;
      return false;
    }
    ++count;
  }
  return count == kExpectedFileConstants;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: relocate_air_paths INPUT_AIR OUTPUT_AIR OLD_PREFIX "
                 "NEW_PREFIX REPORT_JSON\n";
    return 2;
  }
  const std::string input_air = argv[1];
  const std::string output_air = argv[2];
  const std::string old_prefix = argv[3];
  const std::string new_prefix = argv[4];
  const std::string report_path = argv[5];

  ge::Graph source("G4cAttempt69cR2Source");
  if (source.LoadFromFile(input_air) != ge::GRAPH_SUCCESS || !source.IsValid()) {
    return 3;
  }
  std::vector<std::string> source_signature;
  if (!NodeSignature(source, source_signature)) return 4;

  ge::AscendString file_path_attr("file_path");
  size_t rewritten = 0;
  for (const auto &node : source.GetAllNodes()) {
    ge::AscendString type;
    if (node.GetType(type) != ge::GRAPH_SUCCESS) return 5;
    if (ToString(type) != "FileConstant") continue;
    ge::AscendString path;
    if (node.GetAttr(file_path_attr, path) != ge::GRAPH_SUCCESS) return 6;
    const std::string old_path = ToString(path);
    if (!StartsWith(old_path, old_prefix + "/")) {
      std::cerr << "UNEXPECTED_SOURCE_PATH path=" << old_path << std::endl;
      return 7;
    }
    const std::string relative = old_path.substr(old_prefix.size() + 1);
    if (relative.empty() || relative.find('/') != std::string::npos) return 8;
    const std::string new_path = new_prefix + "/" + relative;
    if (!Readable(new_path)) {
      std::cerr << "MISSING_TARGET path=" << new_path << std::endl;
      return 9;
    }
    ge::AscendString updated(new_path.c_str());
    if (node.SetAttr(file_path_attr, updated) != ge::GRAPH_SUCCESS) return 10;
    ++rewritten;
  }
  if (rewritten != kExpectedFileConstants) return 11;
  if (source.SaveToFile(output_air) != ge::GRAPH_SUCCESS) return 12;

  ge::Graph reloaded("G4cAttempt69cR2Reloaded");
  if (reloaded.LoadFromFile(output_air) != ge::GRAPH_SUCCESS ||
      !reloaded.IsValid()) {
    return 13;
  }
  std::vector<std::string> reloaded_signature;
  if (!NodeSignature(reloaded, reloaded_signature)) return 14;
  size_t audited = 0;
  const bool signature_equal = source_signature == reloaded_signature;
  const bool path_audit =
      AuditPaths(reloaded, new_prefix, old_prefix, audited);
  const bool pass = signature_equal && path_audit &&
                    rewritten == kExpectedFileConstants &&
                    audited == kExpectedFileConstants;

  std::ofstream report(report_path, std::ios::trunc);
  if (!report) return 15;
  report << "{\n"
         << "  \"gate\": \"G4c Attempt 69c-r2 relocatable B=4 AIR\",\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"source_node_count\": " << source_signature.size() << ",\n"
         << "  \"reloaded_node_count\": " << reloaded_signature.size()
         << ",\n"
         << "  \"node_signature_equal\": "
         << (signature_equal ? "true" : "false") << ",\n"
         << "  \"rewritten_file_constants\": " << rewritten << ",\n"
         << "  \"audited_file_constants\": " << audited << ",\n"
         << "  \"old_prefix\": \"" << old_prefix << "\",\n"
         << "  \"new_prefix\": \"" << new_prefix << "\",\n"
         << "  \"claim_boundary\": \"AIR path relocation only; native "
            "numerical correctness remains open.\"\n"
         << "}\n";
  report.close();
  std::cout << "G4C_ATTEMPT69C_R2 rewritten=" << rewritten
            << " audited=" << audited
            << " signature_equal=" << signature_equal << std::endl;
  return pass ? 0 : 16;
}
