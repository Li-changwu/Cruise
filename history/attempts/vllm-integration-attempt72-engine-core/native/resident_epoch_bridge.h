#pragma once

#include <cstdint>

extern "C" void *resident_epoch_create(
    const char *air_path, const char *graph_config, const char *func_config,
    const char *external_weight_dir, const char *tiling_path,
    int32_t *status);

extern "C" int32_t resident_epoch_execute(
    void *opaque, int32_t request_count, int32_t max_steps,
    const int64_t *input_token_ids, const int64_t *input_positions,
    const int32_t *input_sequence_lengths, const int32_t *input_eos_token_ids,
    int64_t *output_token_ids, int32_t *output_executed,
    int32_t *output_model_calls, int32_t *output_device_status,
    int32_t *output_feed_calls, int32_t *output_fetch_calls,
    int64_t *output_wall_us);

extern "C" void resident_epoch_destroy(void *opaque);
