#include "resident_device_transfer.h"

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
  decltype(&aclrtMalloc) malloc = nullptr;
  decltype(&aclrtFree) free = nullptr;
  decltype(&aclrtMemset) memset = nullptr;
  decltype(&aclrtMemcpy) memcpy = nullptr;
};

struct DeviceTransferState {
  AclRuntimeApi acl;
  void *payload = nullptr;
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
         ResolveAclSymbol("aclrtMalloc", &api.malloc) &&
         ResolveAclSymbol("aclrtFree", &api.free) &&
         ResolveAclSymbol("aclrtMemset", &api.memset) &&
         ResolveAclSymbol("aclrtMemcpy", &api.memcpy);
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

bool PrepareResidentDeviceIpcPayload(
    const ResidentEpochIpcMetadata *metadata, void **payload_out) {
  if (metadata == nullptr || payload_out == nullptr) return false;
  if (g_state == nullptr) g_state = new DeviceTransferState();
  if (!ResolveAclRuntime(g_state->acl)) return false;
  int32_t current_device = -1;
  if (g_state->acl.get_device(&current_device) != ACL_SUCCESS ||
      current_device != 0) {
    return false;
  }
  if (g_state->payload == nullptr &&
      g_state->acl.malloc(&g_state->payload,
                          CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES,
                          ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    return false;
  }
  if (g_state->acl.memset(g_state->payload,
                          CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES, 0,
                          CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES) != ACL_SUCCESS) {
    return false;
  }
  auto *destination = static_cast<uint8_t *>(g_state->payload);
  for (uint32_t index = 0; index < metadata->segment_count; ++index) {
    const auto &segment = metadata->segments[index];
    void *source = ImportIpcMemory(g_state, segment.key);
    if (source == nullptr) return false;
    const size_t destination_offset =
        static_cast<size_t>(segment.destination_offset);
    const size_t source_offset = static_cast<size_t>(segment.source_offset);
    const size_t copy_bytes = static_cast<size_t>(segment.copy_bytes);
    if (g_state->acl.memcpy(
            destination + destination_offset,
            CRUISE_RESIDENT_IMPORT_PAYLOAD_BYTES - destination_offset,
            static_cast<uint8_t *>(source) + source_offset, copy_bytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE) != ACL_SUCCESS) {
      return false;
    }
  }
  *payload_out = g_state->payload;
  return true;
}

void DestroyResidentDeviceIpcPayload() {
  if (g_state == nullptr) return;
  for (const auto &entry : g_state->ipc_imports) {
    g_state->acl.ipc_close(entry.first.c_str());
  }
  if (g_state->payload != nullptr) g_state->acl.free(g_state->payload);
  delete g_state;
  g_state = nullptr;
}
