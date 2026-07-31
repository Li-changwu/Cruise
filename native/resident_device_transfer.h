#pragma once

#include "resident_epoch_bridge.h"

bool PrepareResidentDeviceIpcPayload(
    const ResidentEpochIpcMetadata *metadata, void **payload_out);

void DestroyResidentDeviceIpcPayload();
