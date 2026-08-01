import struct
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
    ResidentEpochPlan,
    ResidentEpochRequest,
)
import vllm_ascend_resident_epoch.kv_transfer as kv_transfer_module
from vllm_ascend_resident_epoch.kv_transfer import (
    BLOCK_ELEMENTS,
    ELEMENT_BYTES,
    GRAPH_BATCH_SIZE,
    HEADER,
    IPC_EXPORT_BUFFER_BYTES,
    IPC_KEY_BYTES,
    IPC_KEY_COUNT,
    IPC_MAX_SEGMENTS,
    IPC_METADATA_BYTES,
    IPC_METADATA_HEADER,
    IPC_METADATA_VERSION,
    IPC_SEGMENT,
    PAYLOAD_BYTES,
    TRANSFER_HEADER_BYTES,
    DeviceKVSegment,
    DeviceKVTransfer,
    capture_kv_device_transfer,
    capture_kv_snapshot,
    kv_payload_checksum,
    release_kv_device_exports,
)


def test_device_kv_transfer_wire_contract_contains_only_metadata():
    block_bytes = BLOCK_ELEMENTS * ELEMENT_BYTES
    cache_bytes = PAYLOAD_BYTES // 2
    segments = []
    for tensor_index in range(IPC_KEY_COUNT):
        layer = tensor_index // 2
        cache_base = 0 if tensor_index % 2 == 0 else cache_bytes
        for row in (0, 2):
            segments.append(
                DeviceKVSegment(
                    source_offset=0,
                    source_allocation_bytes=block_bytes,
                    destination_offset=cache_base
                    + (layer * GRAPH_BATCH_SIZE + row) * block_bytes,
                    copy_bytes=block_bytes,
                    key=f"{tensor_index:032x}{row:032x}",
                )
            )
    transfer = DeviceKVTransfer(
        transfer_id=7,
        import_mask=0b0101,
        row_generations=(11, 0, 13, 0),
        block_ids=(2, 0, 5, 0),
        segments=tuple(segments),
    )

    wire = transfer.wire_bytes()

    assert len(wire) == IPC_METADATA_BYTES
    assert IPC_METADATA_BYTES < PAYLOAD_BYTES // 100
    assert len(segments) < IPC_MAX_SEGMENTS
    header = IPC_METADATA_HEADER.unpack_from(wire)
    assert header[1] == IPC_METADATA_VERSION
    assert header[3] == len(segments)
    assert header[4] == 0
    first_segment = IPC_SEGMENT.unpack_from(wire, IPC_METADATA_HEADER.size)
    assert first_segment[:4] == (0, block_bytes, 0, block_bytes)
    assert first_segment[4] == segments[0].key.encode()
    used_bytes = IPC_METADATA_HEADER.size + len(segments) * IPC_SEGMENT.size
    assert wire[used_bytes:] == b"\0" * (IPC_METADATA_BYTES - used_bytes)


def test_device_kv_export_uses_storage_base_offsets_and_api_buffer(monkeypatch):
    calls = []
    closes = []
    allocation_ranges = {}

    class FakeStorage:
        def __init__(self, pointer, size):
            self.pointer = pointer
            self.size = size

        def data_ptr(self):
            return self.pointer

        def nbytes(self):
            return self.size

    class FakeTensor:
        dtype = "torch.bfloat16"

        def __init__(self, storage, offset):
            self.storage = storage
            self.offset = offset

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return self.storage.pointer + self.offset

        def numel(self):
            return 8 * BLOCK_ELEMENTS

        def element_size(self):
            return ELEMENT_BYTES

        def untyped_storage(self):
            return self.storage

        def storage_offset(self):
            return self.offset // ELEMENT_BYTES

    class FakeRuntime:
        def ipc_mem_get_export_key(self, pointer, size, buffer_bytes, flag):
            calls.append((pointer, size, buffer_bytes, flag))
            return f"{pointer:064x}", 0

        def ipc_mem_close(self, key):
            closes.append(key)
            return 0

    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(rt=FakeRuntime()))
    monkeypatch.setattr(
        kv_transfer_module,
        "_device_allocation_range",
        lambda pointer: allocation_ranges[pointer],
    )
    kv_caches = []
    for layer in range(28):
        view_bytes = 8 * BLOCK_ELEMENTS * ELEMENT_BYTES
        storage = FakeStorage(0x100000 + layer * 0x400000, 2 * view_bytes)
        key = FakeTensor(storage, 0)
        value = FakeTensor(storage, view_bytes)
        allocation_bytes = storage.size + 2 * BLOCK_ELEMENTS * ELEMENT_BYTES
        allocation_ranges[key.data_ptr()] = (storage.pointer, allocation_bytes)
        allocation_ranges[value.data_ptr()] = (storage.pointer, allocation_bytes)
        kv_caches.append((key, value))
    worker = SimpleNamespace(model_runner=SimpleNamespace(kv_caches=kv_caches))
    plan = ResidentEpochPlan(
        version=CONTRACT_VERSION,
        graph_batch_size=4,
        max_steps=2,
        logical_capacity=8,
        requests=(
            ResidentEpochRequest(
                req_id="prefilled",
                row=0,
                generation=7,
                token_id=42,
                position=3,
                sequence_length=4,
                eos_token_id=151645,
                scheduler_block_ids=(0,),
                device_block_ids=(0, 1),
                state_owner="host",
                kv_import_required=True,
            ),
        ),
        active_mask=(1, 0, 0, 0),
    )

    transfer = capture_kv_device_transfer(worker, plan)

    assert len(calls) == IPC_KEY_COUNT // 2
    assert all(call[2:] == (IPC_EXPORT_BUFFER_BYTES, 1) for call in calls)
    assert len(transfer.segments) == IPC_KEY_COUNT
    assert all(len(segment.key) == IPC_KEY_BYTES for segment in transfer.segments)
    assert all(
        segment.source_allocation_bytes == allocation_bytes
        for segment in transfer.segments
    )
    assert tuple(segment.source_offset for segment in transfer.segments) == (
        0,
        view_bytes,
    ) * (IPC_KEY_COUNT // 2)
    assert all(
        transfer.segments[index].key == transfer.segments[index + 1].key
        for index in range(0, IPC_KEY_COUNT, 2)
    )

    release_kv_device_exports(worker)

    assert len(closes) == IPC_KEY_COUNT // 2


def test_device_kv_signature_describes_the_complete_torch_storage():
    view_bytes = 8 * BLOCK_ELEMENTS * ELEMENT_BYTES

    class FakeStorage:
        def data_ptr(self):
            return 0x100000

        def nbytes(self):
            return 32 * view_bytes

    class FakeTensor:
        def is_contiguous(self):
            return True

        def data_ptr(self):
            return 0x100000 + 8 * view_bytes

        def numel(self):
            return view_bytes // ELEMENT_BYTES

        def element_size(self):
            return ELEMENT_BYTES

        def untyped_storage(self):
            return FakeStorage()

        def storage_offset(self):
            return 8 * view_bytes // ELEMENT_BYTES

    signature = kv_transfer_module._device_tensor_signature(FakeTensor())

    assert signature == (
        0x100000 + 8 * view_bytes,
        view_bytes,
        0x100000,
        32 * view_bytes,
        8 * view_bytes,
    )


def test_device_kv_transfer_splits_blocks_across_cann_allocations(monkeypatch):
    block_bytes = BLOCK_ELEMENTS * ELEMENT_BYTES
    allocation_bytes = 2 * 1024 * 1024
    first_copy_bytes = 49152
    calls = []
    closes = []

    class FakeStorage:
        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

        def nbytes(self):
            return 8 * block_bytes

    class FakeTensor:
        dtype = "torch.bfloat16"

        def __init__(self, pointer):
            self.storage = FakeStorage(pointer)

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return self.storage.pointer

        def numel(self):
            return 8 * BLOCK_ELEMENTS

        def element_size(self):
            return ELEMENT_BYTES

        def untyped_storage(self):
            return self.storage

        def storage_offset(self):
            return 0

    class FakeRuntime:
        def ipc_mem_get_export_key(self, pointer, size, buffer_bytes, flag):
            calls.append((pointer, size, buffer_bytes, flag))
            return f"{pointer:064x}", 0

        def ipc_mem_close(self, key):
            closes.append(key)
            return 0

    key_base = 0x10000000
    value_base = 0x20000000
    key = FakeTensor(key_base + allocation_bytes - first_copy_bytes)
    value = FakeTensor(value_base + allocation_bytes - first_copy_bytes)

    def allocation_range(pointer):
        for base in (key_base, value_base):
            if base <= pointer < base + allocation_bytes:
                return base, allocation_bytes
            if base + allocation_bytes <= pointer < base + 2 * allocation_bytes:
                return base + allocation_bytes, allocation_bytes
        raise AssertionError(f"unexpected device pointer: {pointer}")

    monkeypatch.setitem(sys.modules, "acl", SimpleNamespace(rt=FakeRuntime()))
    monkeypatch.setattr(
        kv_transfer_module, "_device_allocation_range", allocation_range
    )
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
                req_id="boundary",
                row=0,
                generation=7,
                token_id=42,
                position=3,
                sequence_length=4,
                eos_token_id=151645,
                scheduler_block_ids=(0,),
                device_block_ids=(0, 1),
                state_owner="host",
                kv_import_required=True,
            ),
        ),
        active_mask=(1, 0, 0, 0),
    )

    transfer = capture_kv_device_transfer(worker, plan)

    assert len(transfer.segments) == IPC_KEY_COUNT * 2
    assert tuple(segment.copy_bytes for segment in transfer.segments) == (
        first_copy_bytes,
        block_bytes - first_copy_bytes,
    ) * IPC_KEY_COUNT
    assert len(calls) == 4
    assert all(call[1:] == (allocation_bytes, IPC_EXPORT_BUFFER_BYTES, 1) for call in calls)

    release_kv_device_exports(worker)

    assert len(closes) == 4


def test_device_kv_transfer_rejects_gapped_block_segments():
    block_bytes = BLOCK_ELEMENTS * ELEMENT_BYTES
    segments = []
    for tensor_index in range(IPC_KEY_COUNT):
        layer = tensor_index // 2
        cache_base = 0 if tensor_index % 2 == 0 else PAYLOAD_BYTES // 2
        segments.append(
            DeviceKVSegment(
                source_offset=0,
                source_allocation_bytes=block_bytes,
                destination_offset=cache_base + layer * GRAPH_BATCH_SIZE * block_bytes,
                copy_bytes=block_bytes - 2,
                key=f"{tensor_index:064x}",
            )
        )
    transfer = DeviceKVTransfer(
        transfer_id=7,
        import_mask=0b0001,
        row_generations=(11, 0, 0, 0),
        block_ids=(2, 0, 0, 0),
        segments=tuple(segments),
    )

    with pytest.raises(ValueError, match="coverage is incomplete"):
        transfer.validate()


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


def test_capture_stock_paged_kv_imports_multiple_scheduler_blocks_by_row():
    key = torch.zeros((4, 128, 4, 128), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    for block in range(4):
        key[block].fill_(block + 1)
        value[block].fill_(block + 5)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(kv_caches=[(key, value)] * 28)
    )

    requests = []
    scheduler_blocks = {0: 2, 2: 0, 3: 3}
    for row, scheduler_block in scheduler_blocks.items():
        requests.append(
            ResidentEpochRequest(
                req_id=f"r{row}",
                row=row,
                generation=row + 10,
                token_id=42 + row,
                position=3 + row,
                sequence_length=4 + row,
                eos_token_id=151645,
                scheduler_block_ids=(scheduler_block,),
                device_block_ids=(row * 2, row * 2 + 1),
                state_owner="host",
                kv_import_required=True,
            )
        )
    plan = ResidentEpochPlan(
        version=CONTRACT_VERSION,
        graph_batch_size=4,
        max_steps=1,
        logical_capacity=8,
        requests=tuple(requests),
        active_mask=(1, 0, 1, 1),
    )

    snapshot = capture_kv_snapshot(worker, plan)

    assert snapshot.import_mask == 0b1101
    assert snapshot.row_generations == (10, 0, 12, 13)
    row_bytes = BLOCK_ELEMENTS * ELEMENT_BYTES
    expected_key_words = {0: 0x4040, 1: 0, 2: 0x3F80, 3: 0x4080}
    for row, expected in expected_key_words.items():
        actual = struct.unpack_from("<H", snapshot.payload, row * row_bytes)[0]
        assert actual == expected
