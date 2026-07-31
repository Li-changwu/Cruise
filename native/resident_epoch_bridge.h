#pragma once

#include <cstdint>

#include "resident_epoch_protocol.h"

#pragma pack(push, 1)
struct ResidentEpochIpcMetadata {
  uint64_t magic;
  uint32_t version;
  uint32_t import_mask;
  uint64_t source_bytes;
  int32_t row_generations[4];
  int32_t block_ids[4];
  uint64_t source_offsets[56];
  char keys[56][64];
};
#pragma pack(pop)

static_assert(sizeof(ResidentEpochIpcMetadata) ==
                  CRUISE_RESIDENT_IPC_METADATA_BYTES,
              "resident Device IPC metadata ABI changed");

extern "C" void *resident_epoch_create(
    const char *air_path, const char *graph_config, const char *func_config,
    const char *external_weight_dir, const char *tiling_path,
    int32_t *status);

extern "C" int32_t resident_epoch_execute(
    void *opaque, int32_t request_count, int32_t max_steps,
    const int64_t *input_token_ids, const int64_t *input_positions,
    const int32_t *input_sequence_lengths, const int32_t *input_eos_token_ids,
    const int32_t *input_row_generations,
    int64_t *output_token_ids, int32_t *output_executed,
    int32_t *output_row_generations,
    int32_t *output_model_calls, int32_t *output_device_status,
    int32_t *output_feed_calls, int32_t *output_fetch_calls,
    int32_t *output_commit_state, int32_t *output_kv_import_checksum,
    int64_t *output_wall_us, int64_t *output_native_cpu_us,
    int64_t *output_declared_input_bytes,
    int64_t *output_declared_output_bytes,
    const char *transfer_path, uint64_t transfer_id,
    const ResidentEpochIpcMetadata *ipc_metadata);

extern "C" void resident_epoch_destroy(void *opaque);
