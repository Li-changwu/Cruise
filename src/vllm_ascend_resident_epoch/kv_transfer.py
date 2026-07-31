"""Paged-KV transfer contracts for the resident epoch backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
import secrets
import struct
from typing import Any
import zlib


TRANSFER_MAGIC = 0x4352554953454B56  # "CRUISEKV"
TRANSFER_VERSION = 1
TRANSFER_HEADER_BYTES = 80
IPC_METADATA_MAGIC = 0x4352554953454950  # "CRUISEIP"
IPC_METADATA_VERSION = 3
IPC_KEY_BYTES = 64
IPC_EXPORT_BUFFER_BYTES = 128
LAYERS = 28
GRAPH_BATCH_SIZE = 4
BLOCK_SIZE = 128
NUM_KV_HEADS = 4
HEAD_SIZE = 128
ELEMENT_BYTES = 2
BLOCK_ELEMENTS = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
ROW_BYTES = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * ELEMENT_BYTES
PAYLOAD_BYTES = LAYERS * GRAPH_BATCH_SIZE * ROW_BYTES
HEADER = struct.Struct("<QIIQQI4i7I")
IPC_KEY_COUNT = LAYERS * 2
IPC_MAX_SEGMENTS_PER_BLOCK = 2
IPC_MAX_SEGMENTS = (
    IPC_KEY_COUNT * GRAPH_BATCH_SIZE * IPC_MAX_SEGMENTS_PER_BLOCK
)
IPC_METADATA_HEADER = struct.Struct("<QIIII4i4i")
IPC_SEGMENT = struct.Struct(f"<QQQQ{IPC_KEY_BYTES}s")
IPC_METADATA_BYTES = IPC_METADATA_HEADER.size + IPC_MAX_SEGMENTS * IPC_SEGMENT.size


@dataclass(frozen=True)
class ResidentKVSnapshot:
    transfer_id: int
    import_mask: int
    row_generations: tuple[int, ...]
    payload: bytes
    checksum: int

    def validate(self) -> None:
        if self.transfer_id <= 0:
            raise ValueError("KV transfer id must be positive")
        if len(self.row_generations) != GRAPH_BATCH_SIZE:
            raise ValueError("KV transfer generation vector must have four rows")
        if any(generation < 0 for generation in self.row_generations):
            raise ValueError("KV transfer generations must be non-negative")
        if self.import_mask <= 0 or self.import_mask >= 1 << GRAPH_BATCH_SIZE:
            raise ValueError("KV transfer mask must select at least one row")
        for row, generation in enumerate(self.row_generations):
            selected = bool(self.import_mask & (1 << row))
            if selected != (generation > 0):
                raise ValueError("KV transfer mask and generations disagree")
        if len(self.payload) != PAYLOAD_BYTES:
            raise ValueError(
                f"KV transfer payload must be {PAYLOAD_BYTES} bytes, "
                f"got {len(self.payload)}"
            )
        if self.checksum != kv_payload_checksum(self.payload, self.import_mask):
            raise ValueError("KV transfer checksum does not match its payload")


@dataclass(frozen=True)
class DeviceKVSegment:
    source_offset: int
    source_allocation_bytes: int
    destination_offset: int
    copy_bytes: int
    key: str

    def validate(self) -> None:
        if (
            self.source_offset < 0
            or self.source_allocation_bytes <= 0
            or self.copy_bytes <= 0
            or self.source_offset + self.copy_bytes
            > self.source_allocation_bytes
        ):
            raise ValueError("device KV source segment is outside its allocation")
        if (
            self.destination_offset < 0
            or self.destination_offset + self.copy_bytes > PAYLOAD_BYTES
        ):
            raise ValueError("device KV destination segment is outside the payload")
        encoded = self.key.encode("ascii")
        if not encoded or len(encoded) > IPC_KEY_BYTES:
            raise ValueError("device KV IPC key does not fit the wire contract")

    def wire_bytes(self) -> bytes:
        self.validate()
        return IPC_SEGMENT.pack(
            self.source_offset,
            self.source_allocation_bytes,
            self.destination_offset,
            self.copy_bytes,
            self.key.encode("ascii").ljust(IPC_KEY_BYTES, b"\0"),
        )


@dataclass(frozen=True)
class DeviceKVTransfer:
    """Metadata for importing stock NPU KV blocks without a Host payload."""

    transfer_id: int
    import_mask: int
    row_generations: tuple[int, ...]
    block_ids: tuple[int, ...]
    segments: tuple[DeviceKVSegment, ...]

    def validate(self) -> None:
        if self.transfer_id <= 0:
            raise ValueError("device KV transfer id must be positive")
        if len(self.row_generations) != GRAPH_BATCH_SIZE:
            raise ValueError("device KV transfer generation vector must have four rows")
        if len(self.block_ids) != GRAPH_BATCH_SIZE:
            raise ValueError("device KV transfer block vector must have four rows")
        if self.import_mask <= 0 or self.import_mask >= 1 << GRAPH_BATCH_SIZE:
            raise ValueError("device KV transfer mask must select at least one row")
        for row in range(GRAPH_BATCH_SIZE):
            selected = bool(self.import_mask & (1 << row))
            generation = self.row_generations[row]
            block_id = self.block_ids[row]
            if selected:
                if generation <= 0 or block_id < 0:
                    raise ValueError("selected device KV rows need generation and block")
            elif generation != 0 or block_id != 0:
                raise ValueError("unselected device KV rows must be zeroed")
        if not self.segments or len(self.segments) > IPC_MAX_SEGMENTS:
            raise ValueError("device KV transfer segment count is outside the ABI")
        intervals: list[tuple[int, int]] = []
        block_intervals: dict[int, list[tuple[int, int]]] = {}
        total_bytes = 0
        block_bytes = BLOCK_ELEMENTS * ELEMENT_BYTES
        cache_bytes = PAYLOAD_BYTES // 2
        for segment in self.segments:
            segment.validate()
            cache_offset = segment.destination_offset % cache_bytes
            layer_row = cache_offset // block_bytes
            row = layer_row % GRAPH_BATCH_SIZE
            block_offset = cache_offset % block_bytes
            if not self.import_mask & (1 << row):
                raise ValueError("device KV segment targets an unselected row")
            if block_offset + segment.copy_bytes > block_bytes:
                raise ValueError("device KV segment crosses a destination block")
            start = segment.destination_offset
            end = start + segment.copy_bytes
            intervals.append((start, end))
            block_start = segment.destination_offset - block_offset
            block_intervals.setdefault(block_start, []).append((start, end))
            total_bytes += segment.copy_bytes
        intervals.sort()
        if any(
            current[0] < previous[1]
            for previous, current in zip(intervals, intervals[1:])
        ):
            raise ValueError("device KV destination segments overlap")
        selected_rows = self.import_mask.bit_count()
        expected_blocks = selected_rows * IPC_KEY_COUNT
        if len(block_intervals) != expected_blocks:
            raise ValueError("device KV segments do not cover every selected block")
        for block_start, spans in block_intervals.items():
            if len(spans) > IPC_MAX_SEGMENTS_PER_BLOCK:
                raise ValueError("device KV block has too many allocation segments")
            spans.sort()
            cursor = block_start
            for start, end in spans:
                if start != cursor:
                    raise ValueError("device KV block segments have a coverage gap")
                cursor = end
            if cursor != block_start + block_bytes:
                raise ValueError("device KV block coverage is incomplete")
        if total_bytes != selected_rows * IPC_KEY_COUNT * block_bytes:
            raise ValueError("device KV segments do not cover every selected block")

    def wire_bytes(self) -> bytes:
        self.validate()
        header = IPC_METADATA_HEADER.pack(
            IPC_METADATA_MAGIC,
            IPC_METADATA_VERSION,
            self.import_mask,
            len(self.segments),
            0,
            *self.row_generations,
            *self.block_ids,
        )
        segment_bytes = b"".join(segment.wire_bytes() for segment in self.segments)
        payload = (header + segment_bytes).ljust(IPC_METADATA_BYTES, b"\0")
        if len(payload) != IPC_METADATA_BYTES:
            raise AssertionError("device KV IPC metadata ABI size changed")
        return payload


@dataclass
class _DeviceKVExportTable:
    signature: tuple[tuple[int, int, int, int, int], ...]
    view_bytes: int
    exports: dict[tuple[int, int], str] = field(default_factory=dict)

    def export_range(self, allocation_ptr: int, allocation_bytes: int) -> str:
        allocation = (allocation_ptr, allocation_bytes)
        existing = self.exports.get(allocation)
        if existing is not None:
            return existing
        import acl

        key, status = acl.rt.ipc_mem_get_export_key(
            allocation_ptr,
            allocation_bytes,
            IPC_EXPORT_BUFFER_BYTES,
            1,
        )
        if status != 0 or not isinstance(key, str) or not key:
            self.close()
            raise RuntimeError(
                "failed to export stock KV allocation range, "
                f"status={status}, allocation_bytes={allocation_bytes}"
            )
        self.exports[allocation] = key
        return key

    def close(self) -> None:
        try:
            import acl
        except Exception:
            return
        for key in dict.fromkeys(self.exports.values()):
            try:
                acl.rt.ipc_mem_close(key)
            except Exception:
                pass
        self.exports.clear()


def kv_payload_checksum(payload: bytes, import_mask: int) -> int:
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError("cannot checksum an incomplete KV transfer payload")
    block_bytes = BLOCK_ELEMENTS * ELEMENT_BYTES
    cache_bytes = LAYERS * GRAPH_BATCH_SIZE * block_bytes
    checksum = 1
    for cache_base in (0, cache_bytes):
        for layer in range(LAYERS):
            for row in range(GRAPH_BATCH_SIZE):
                if not import_mask & (1 << row):
                    continue
                offset = cache_base + (layer * GRAPH_BATCH_SIZE + row) * block_bytes
                checksum = zlib.adler32(
                    payload[offset : offset + block_bytes], checksum
                )
    return checksum & 0xFFFFFFFF


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("KV cache entry is not a torch.Tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"unsupported KV cache dtype: {tensor.dtype}")
    cpu = tensor.detach().to(device="cpu").contiguous()
    # NumPy has no stable bfloat16 ABI on all supported versions. Viewing the
    # CPU storage as uint16 preserves the exact device cache bytes.
    return cpu.view(torch.uint16).numpy().tobytes()


@lru_cache(maxsize=1)
def _acl_address_range_api() -> tuple[Any, Any]:
    import ctypes

    try:
        library = ctypes.CDLL("libascendcl.so")
        function = library.aclrtMemGetAddressRange
    except (OSError, AttributeError) as exc:
        raise RuntimeError(
            "CANN does not provide the required aclrtMemGetAddressRange API"
        ) from exc
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    function.restype = ctypes.c_int
    return library, function


def _device_allocation_range(data_ptr: int) -> tuple[int, int]:
    import ctypes

    _, function = _acl_address_range_api()
    base = ctypes.c_void_p()
    size = ctypes.c_size_t()
    status = int(
        function(
            ctypes.c_void_p(data_ptr),
            ctypes.byref(base),
            ctypes.byref(size),
        )
    )
    allocation_ptr = int(base.value or 0)
    allocation_bytes = int(size.value)
    if status != 0 or allocation_ptr <= 0 or allocation_bytes <= 0:
        raise RuntimeError(
            "failed to resolve the CANN allocation containing stock KV, "
            f"status={status}"
        )
    return allocation_ptr, allocation_bytes


def _device_tensor_signature(tensor: Any) -> tuple[int, int, int, int, int]:
    if not getattr(tensor, "is_contiguous", lambda: False)():
        raise RuntimeError("stock KV cache entries must be contiguous for IPC export")
    data_ptr = int(tensor.data_ptr())
    view_bytes = int(tensor.numel()) * int(tensor.element_size())
    storage = tensor.untyped_storage()
    storage_ptr = int(storage.data_ptr())
    storage_bytes = int(storage.nbytes())
    if (
        data_ptr <= 0
        or view_bytes <= 0
        or storage_ptr <= 0
        or storage_bytes <= 0
    ):
        raise RuntimeError(
            "stock KV cache entry has no exportable device storage: "
            f"data_ptr={data_ptr}, view_bytes={view_bytes}, "
            f"storage_ptr={storage_ptr}, storage_bytes={storage_bytes}"
        )
    storage_offset_bytes = int(tensor.storage_offset()) * int(tensor.element_size())
    if data_ptr - storage_ptr != storage_offset_bytes:
        raise RuntimeError("stock KV cache storage offset does not match its device pointer")
    return data_ptr, view_bytes, storage_ptr, storage_bytes, storage_offset_bytes


def _get_device_kv_exports(worker: Any, kv_caches: list[Any]) -> _DeviceKVExportTable:
    """Export each persistent stock KV allocation once per worker lifetime."""

    signature: list[tuple[int, int, int, int, int]] = []
    for layer_cache in kv_caches:
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
            raise RuntimeError("stock vLLM KV cache layer is not a K/V pair")
        for tensor in layer_cache[:2]:
            if str(getattr(tensor, "dtype", None)) != "torch.bfloat16":
                raise ValueError("device KV IPC currently requires bfloat16 caches")
            signature.append(_device_tensor_signature(tensor))

    view_bytes = signature[0][1]
    if any(entry[1] != view_bytes for entry in signature):
        raise RuntimeError("key/value KV allocations do not have a common size")
    current = getattr(worker, "_resident_epoch_device_kv_exports", None)
    if isinstance(current, _DeviceKVExportTable) and current.signature == tuple(signature):
        return current
    if isinstance(current, _DeviceKVExportTable):
        current.close()
    table = _DeviceKVExportTable(
        signature=tuple(signature),
        view_bytes=view_bytes,
    )
    worker._resident_epoch_device_kv_exports = table
    return table


def capture_kv_device_transfer(worker: Any, plan: Any) -> DeviceKVTransfer:
    """Create IPC metadata while leaving every KV byte on the NPU."""

    requests = [request for request in plan.requests if request.kv_import_required]
    if not requests:
        raise ValueError("device KV transfer requested for a plan without imports")
    runner = getattr(worker, "model_runner", None)
    kv_caches = getattr(runner, "kv_caches", None)
    if not isinstance(kv_caches, list) or len(kv_caches) != LAYERS:
        raise RuntimeError("stock vLLM KV cache layout is not the M4b layout")
    table = _get_device_kv_exports(worker, kv_caches)
    block_ids = [0] * GRAPH_BATCH_SIZE
    generations = [0] * GRAPH_BATCH_SIZE
    import_mask = 0
    for request in requests:
        if len(request.scheduler_block_ids) != 1:
            raise ValueError("device KV IPC requires one scheduler block per row")
        block_ids[request.row] = int(request.scheduler_block_ids[0])
        generations[request.row] = int(request.generation)
        import_mask |= 1 << request.row
    block_bytes = BLOCK_ELEMENTS * ELEMENT_BYTES
    cache_bytes = PAYLOAD_BYTES // 2
    segments: list[DeviceKVSegment] = []
    tensors = [tensor for layer_cache in kv_caches for tensor in layer_cache[:2]]
    try:
        for tensor_index, tensor in enumerate(tensors):
            layer = tensor_index // 2
            cache_base = 0 if tensor_index % 2 == 0 else cache_bytes
            tensor_ptr = int(tensor.data_ptr())
            for request in requests:
                block_id = int(request.scheduler_block_ids[0])
                block_start = tensor_ptr + block_id * block_bytes
                block_end = block_start + block_bytes
                if block_id < 0 or block_end > tensor_ptr + table.view_bytes:
                    raise ValueError("scheduler block is outside the stock KV view")
                destination_start = cache_base + (
                    layer * GRAPH_BATCH_SIZE + request.row
                ) * block_bytes
                cursor = block_start
                copied = 0
                segment_count = 0
                while cursor < block_end:
                    allocation_ptr, allocation_bytes = _device_allocation_range(cursor)
                    allocation_end = allocation_ptr + allocation_bytes
                    if cursor < allocation_ptr or cursor >= allocation_end:
                        raise RuntimeError(
                            "CANN allocation range does not contain the KV cursor"
                        )
                    copy_bytes = min(block_end, allocation_end) - cursor
                    if copy_bytes <= 0:
                        raise RuntimeError("CANN allocation range made no KV progress")
                    segment_count += 1
                    if segment_count > IPC_MAX_SEGMENTS_PER_BLOCK:
                        raise RuntimeError(
                            "one KV block spans more CANN allocations than the ABI allows"
                        )
                    key = table.export_range(allocation_ptr, allocation_bytes)
                    segments.append(
                        DeviceKVSegment(
                            source_offset=cursor - allocation_ptr,
                            source_allocation_bytes=allocation_bytes,
                            destination_offset=destination_start + copied,
                            copy_bytes=copy_bytes,
                            key=key,
                        )
                    )
                    cursor += copy_bytes
                    copied += copy_bytes
        transfer = DeviceKVTransfer(
            transfer_id=secrets.randbits(63) or 1,
            import_mask=import_mask,
            row_generations=tuple(generations),
            block_ids=tuple(block_ids),
            segments=tuple(segments),
        )
        transfer.validate()
        return transfer
    except Exception:
        table.close()
        worker._resident_epoch_device_kv_exports = None
        raise


def release_kv_device_exports(worker: Any) -> None:
    exports = getattr(worker, "_resident_epoch_device_kv_exports", None)
    if isinstance(exports, _DeviceKVExportTable):
        exports.close()
    worker._resident_epoch_device_kv_exports = None


def capture_kv_snapshot(worker: Any, plan: Any) -> ResidentKVSnapshot:
    """Copy imported scheduler blocks into a compact, row-major snapshot."""

    import torch

    requests = [request for request in plan.requests if request.kv_import_required]
    if not requests:
        raise ValueError("KV snapshot requested for a plan without imports")
    if plan.logical_capacity > BLOCK_SIZE:
        raise ValueError("M1 transfer currently supports one 128-token block")

    by_row = {request.row: request for request in requests}
    runner = getattr(worker, "model_runner", None)
    kv_caches = getattr(runner, "kv_caches", None)
    if not isinstance(kv_caches, list) or len(kv_caches) != LAYERS:
        raise RuntimeError("stock vLLM KV cache layout is not the M1 layout")

    zero_row = torch.zeros(
        (BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE),
        dtype=torch.bfloat16,
        device="cpu",
    )
    key_layers: list[torch.Tensor] = []
    value_layers: list[torch.Tensor] = []
    for layer_cache in kv_caches:
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
            raise RuntimeError("stock vLLM KV cache layer is not a K/V pair")
        key_cache, value_cache = layer_cache[0], layer_cache[1]
        key_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        for row in range(GRAPH_BATCH_SIZE):
            request = by_row.get(row)
            if request is None:
                key_rows.append(zero_row.clone())
                value_rows.append(zero_row.clone())
                continue
            if len(request.scheduler_block_ids) != 1:
                raise ValueError("M1 transfer requires one scheduler block per row")
            block_id = request.scheduler_block_ids[0]
            if block_id < 0 or block_id >= key_cache.shape[0]:
                raise ValueError("scheduler block is outside the stock KV cache")
            key_block = key_cache[block_id]
            value_block = value_cache[block_id]
            expected = (BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
            if tuple(key_block.shape) != expected or tuple(value_block.shape) != expected:
                raise RuntimeError("stock vLLM KV block shape differs from M1 layout")
            key_rows.append(key_block.detach().to(device="cpu").contiguous())
            value_rows.append(value_block.detach().to(device="cpu").contiguous())
        key_layers.append(torch.stack(key_rows, dim=0))
        value_layers.append(torch.stack(value_rows, dim=0))

    key_payload = _tensor_bytes(torch.stack(key_layers, dim=0))
    value_payload = _tensor_bytes(torch.stack(value_layers, dim=0))
    import_generations = [0] * GRAPH_BATCH_SIZE
    for request in requests:
        import_generations[request.row] = request.generation
    payload = key_payload + value_payload
    import_mask = sum(1 << request.row for request in requests)
    snapshot = ResidentKVSnapshot(
        transfer_id=secrets.randbits(63) or 1,
        import_mask=import_mask,
        row_generations=tuple(import_generations),
        payload=payload,
        checksum=kv_payload_checksum(payload, import_mask),
    )
    snapshot.validate()
    return snapshot


def write_kv_snapshot(path: str | Path, snapshot: ResidentKVSnapshot) -> None:
    snapshot.validate()
    destination = Path(path).resolve()
    if not str(destination).startswith("/dev/shm/"):
        raise ValueError("KV transfer files must be under /dev/shm")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{snapshot.transfer_id}"
    )
    header = HEADER.pack(
        TRANSFER_MAGIC,
        TRANSFER_VERSION,
        TRANSFER_HEADER_BYTES,
        snapshot.transfer_id,
        len(snapshot.payload),
        snapshot.import_mask,
        *snapshot.row_generations,
        LAYERS,
        GRAPH_BATCH_SIZE,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE,
        ELEMENT_BYTES,
        snapshot.checksum,
    )
    if temporary.exists():
        raise RuntimeError(f"stale KV transfer temporary file exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(header)
            stream.write(snapshot.payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
