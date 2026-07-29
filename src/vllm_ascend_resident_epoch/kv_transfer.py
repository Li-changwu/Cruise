"""Bounded one-shot Host-to-device Paged-KV transfer for M1."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import struct
from typing import Any
import zlib


TRANSFER_MAGIC = 0x4352554953454B56  # "CRUISEKV"
TRANSFER_VERSION = 1
TRANSFER_HEADER_BYTES = 80
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
