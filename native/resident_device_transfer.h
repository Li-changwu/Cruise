#pragma once

#include <cstdint>

#include "resident_epoch_bridge.h"

extern "C" __attribute__((visibility("default"))) int32_t
resident_device_transfer_prepare(const ResidentEpochIpcMetadata *metadata,
                                 void *destination,
                                 size_t destination_bytes);

extern "C" __attribute__((visibility("default"))) void
resident_device_transfer_destroy();
