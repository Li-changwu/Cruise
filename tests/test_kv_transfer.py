import struct
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
    ResidentEpochPlan,
    ResidentEpochRequest,
)
from vllm_ascend_resident_epoch.kv_transfer import (
    BLOCK_ELEMENTS,
    ELEMENT_BYTES,
    GRAPH_BATCH_SIZE,
    HEADER,
    PAYLOAD_BYTES,
    TRANSFER_HEADER_BYTES,
    capture_kv_snapshot,
    kv_payload_checksum,
)


def test_capture_stock_paged_kv_uses_scheduler_block_and_resident_row():
    key = torch.zeros((2, 128, 4, 128), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    key[1].fill_(1)
    value[1].fill_(2)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(kv_caches=[(key, value)] * 28)
    )
    plan = ResidentEpochPlan(
        version=CONTRACT_VERSION,
        graph_batch_size=4,
        max_steps=2,
        logical_capacity=8,
        requests=(
            ResidentEpochRequest(
                req_id="prefilled",
                row=1,
                generation=7,
                token_id=42,
                position=3,
                sequence_length=4,
                eos_token_id=151645,
                scheduler_block_ids=(1,),
                device_block_ids=(2, 3),
                state_owner="host",
                kv_import_required=True,
            ),
        ),
        active_mask=(0, 1, 0, 0),
    )

    snapshot = capture_kv_snapshot(worker, plan)
    assert HEADER.size == TRANSFER_HEADER_BYTES == 80
    assert len(snapshot.payload) == PAYLOAD_BYTES
    assert snapshot.import_mask == 0b0010
    assert snapshot.row_generations == (0, 7, 0, 0)
    assert snapshot.checksum == kv_payload_checksum(snapshot.payload, 0b0010)

    row_bytes = BLOCK_ELEMENTS * ELEMENT_BYTES
    imported_key_offset = row_bytes
    value_base = 28 * GRAPH_BATCH_SIZE * row_bytes
    assert struct.unpack_from("<H", snapshot.payload, imported_key_offset)[0] == 0x3F80
    assert struct.unpack_from(
        "<H", snapshot.payload, value_base + imported_key_offset
    )[0] == 0x4000
    assert struct.unpack_from("<H", snapshot.payload, 0)[0] == 0
