#include "resident_device_transfer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <string>

#include <dlfcn.h>

#include "acl/acl_rt.h"
#include "resident_epoch_protocol.h"

namespace {
struct AclRuntimeApi {
  decltype(&aclrtGetDevice) get_device = nullptr;
  decltype(&aclrtIpcMemImportByKey) ipc_import = nullptr;
  decltype(&aclrtIpcMemClose) ipc_close = nullptr;
  decltype(&aclrtMemcpy) memcpy = nullptr;
};

struct DeviceTransferState {
  AclRuntimeApi acl;
  std::map<std::string, void *> ipc_imports;
};

DeviceTransferState *g_state = nullptr;

bool IsLibAclRtSymbol(void *symbol) {
  if (symbol == nullptr) return false;
  Dl_info info{};
  if (dladdr(symbol, &info) == 0 || info.dli_fname == nullptr) return false;
  const char *filename = std::strrchr(info.dli_fname, '/');
  filename = filename == nullptr ? info.dli_fname : filename + 1;
  constexpr char kLibraryName[] = "libacl_rt.so";
  return std::strncmp(filename, kLibraryName, sizeof(kLibraryName) - 1) == 0 &&
         (filename[sizeof(kLibraryName) - 1] == '\0' ||
          filename[sizeof(kLibraryName) - 1] == '.');
}

template <typename Function>
bool ResolveAclSymbol(const char *name, Function *function) {
  if (name == nullptr || function == nullptr) return false;
  void *symbol = dlsym(RTLD_DEFAULT, name);
  if (!IsLibAclRtSymbol(symbol)) return false;
  *function = reinterpret_cast<Function>(symbol);
  return true;
}

bool ResolveAclRuntime(AclRuntimeApi &api) {
  return ResolveAclSymbol("aclrtGetDevice", &api.get_device) &&
         ResolveAclSymbol("aclrtIpcMemImportByKey", &api.ipc_import) &&
         ResolveAclSymbol("aclrtIpcMemClose", &api.ipc_close) &&
         ResolveAclSymbol("aclrtMemcpy", &api.memcpy);
}

bool IsIpcKey(const char *key) {
  if (key == nullptr) return false;
  const size_t key_length = strnlen(key, CRUISE_RESIDENT_IPC_KEY_BYTES);
  return key_length != 0 &&
         (key_length == CRUISE_RESIDENT_IPC_KEY_BYTES ||
          std::all_of(key + key_length + 1,
                      key + CRUISE_RESIDENT_IPC_KEY_BYTES,
                      [](char value) { return value == '\0'; }));
}

void *ImportIpcMemory(DeviceTransferState *state, const char *key) {
  if (state == nullptr || key == nullptr || *key == '\0') return nullptr;
  const size_t key_length = strnlen(key, CRUISE_RESIDENT_IPC_KEY_BYTES);
  if (key_length == 0) return nullptr;
  const std::string key_string(key, key_length);
  const auto existing = state->ipc_imports.find(key_string);
  if (existing != state->ipc_imports.end()) return existing->second;
  std::array<char, CRUISE_RESIDENT_IPC_KEY_BYTES + 1> terminated_key{};
  std::memcpy(terminated_key.data(), key, key_length);
  void *device_ptr = nullptr;
  const auto status = state->acl.ipc_import(
      &device_ptr, terminated_key.data(), ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT);
  if (status != ACL_SUCCESS || device_ptr == nullptr) return nullptr;
  state->ipc_imports.emplace(key_string, device_ptr);
  return device_ptr;
}
}  // namespace

extern "C" int32_t resident_device_transfer_prepare(
    const ResidentEpochIpcMetadata *metadata, void *destination_payload,
    size_t destination_bytes) {
  if (metadata == nullptr || destination_payload == nullptr ||
      destination_bytes != CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES ||
      metadata->magic != CRUISE_RESIDENT_IPC_METADATA_MAGIC ||
      metadata->version != CRUISE_RESIDENT_IPC_METADATA_VERSION ||
      metadata->segment_count == 0 ||
      metadata->segment_count > CRUISE_RESIDENT_IPC_MAX_SEGMENTS) {
    return 1;
  }
  if (g_state == nullptr) g_state = new DeviceTransferState();
  if (!ResolveAclRuntime(g_state->acl)) return 2;
  int32_t current_device = -1;
  if (g_state->acl.get_device(&current_device) != ACL_SUCCESS ||
      current_device != 0) {
    return 3;
  }
  // Metadata validation guarantees full coverage for every selected row;
  // unselected rows are never read by the import controller.
  auto *destination = static_cast<uint8_t *>(destination_payload);
  for (uint32_t index = 0; index < metadata->segment_count; ++index) {
    const auto &segment = metadata->segments[index];
    if (!IsIpcKey(segment.key) || segment.source_allocation_bytes == 0 ||
        segment.copy_bytes == 0 ||
        segment.source_offset > segment.source_allocation_bytes ||
        segment.copy_bytes >
            segment.source_allocation_bytes - segment.source_offset ||
        segment.destination_offset > CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES ||
        segment.copy_bytes > CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES -
                                 segment.destination_offset) {
      return 4;
    }
    void *source = ImportIpcMemory(g_state, segment.key);
    if (source == nullptr) return 5;
    const size_t destination_offset =
        static_cast<size_t>(segment.destination_offset);
    const size_t source_offset = static_cast<size_t>(segment.source_offset);
    const size_t copy_bytes = static_cast<size_t>(segment.copy_bytes);
    if (g_state->acl.memcpy(
            destination + destination_offset,
            CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES - destination_offset,
            static_cast<uint8_t *>(source) + source_offset, copy_bytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE) != ACL_SUCCESS) {
      return 6;
    }
  }
  return 0;
}

extern "C" void resident_device_transfer_destroy() {
  if (g_state == nullptr) return;
  for (const auto &entry : g_state->ipc_imports) {
    g_state->acl.ipc_close(entry.first.c_str());
  }
  delete g_state;
  g_state = nullptr;
}
