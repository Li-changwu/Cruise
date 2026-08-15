#!/usr/bin/env python3
"""Export a weight-free B4/K384 in-place Paged-KV update probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch_npu


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from inspect_kv_alias import inspect_graph  # noqa: E402


BATCH_SIZE = 4
KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 128
BLOCKS_PER_ROW = 3
PHYSICAL_BLOCKS = BATCH_SIZE * BLOCKS_PER_ROW
PACK_WIDTH = 16
CACHE_SHAPE = (
    PHYSICAL_BLOCKS,
    KV_HEADS * HEAD_DIM // PACK_WIDTH,
    BLOCK_SIZE,
    PACK_WIDTH,
)
SLOTS = tuple(row * BLOCKS_PER_ROW * BLOCK_SIZE for row in range(BATCH_SIZE))


class KvAliasProbe(torch.nn.Module):
    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch_npu.npu_scatter_pa_kv_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            cache_mode="PA_NZ",
        )
        return key_cache, value_cache


def probe_inputs() -> tuple[torch.Tensor, ...]:
    key_rows = [
        torch.full((KV_HEADS, HEAD_DIM), row + 1, dtype=torch.bfloat16)
        for row in range(BATCH_SIZE)
    ]
    value_rows = [
        torch.full((KV_HEADS, HEAD_DIM), -(row + 1), dtype=torch.bfloat16)
        for row in range(BATCH_SIZE)
    ]
    return (
        torch.stack(key_rows).npu(),
        torch.stack(value_rows).npu(),
        torch.zeros(CACHE_SHAPE, dtype=torch.bfloat16, device="npu"),
        torch.zeros(CACHE_SHAPE, dtype=torch.bfloat16, device="npu"),
        torch.tensor(SLOTS, dtype=torch.int32, device="npu"),
    )


def eager_exact(model: KvAliasProbe) -> bool:
    key, value, key_cache, value_cache, slots = probe_inputs()
    with torch.no_grad():
        actual_key, actual_value = model(key, value, key_cache, value_cache, slots)
        torch.npu.synchronize()
    expected_key = torch.zeros_like(actual_key)
    expected_value = torch.zeros_like(actual_value)
    for row, slot in enumerate(SLOTS):
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        expected_key[block, :, offset, :] = key[row].reshape(-1, PACK_WIDTH)
        expected_value[block, :, offset, :] = value[row].reshape(-1, PACK_WIDTH)
    exact = torch.equal(actual_key, expected_key) and torch.equal(
        actual_value, expected_value
    )
    del key, value, key_cache, value_cache, slots
    del actual_key, actual_value, expected_key, expected_value
    return exact


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    torch.npu.set_device(0)
    model = KvAliasProbe().eval().npu()
    exact = eager_exact(model)
    sample_inputs = probe_inputs()

    from torchair.configs.compiler_config import CompilerConfig
    from torchair.npu_export import dynamo_export

    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        *sample_inputs,
        model=model,
        export_path=str(args.output_dir),
        export_name="kv_alias_probe",
        dynamic=False,
        config=config,
    )
    torch.npu.synchronize()
    air = args.output_dir / "kv_alias_probe.air"
    graph = args.output_dir / "dynamo.pbtxt"
    structure = inspect_graph(graph) if graph.is_file() else {
        "pass": False,
        "error": "dynamo.pbtxt was not exported",
    }
    (args.output_dir / "graph-structure.json").write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "gate": "V2-KV-ALIAS-EXPORT",
        "pass": bool(exact and air.is_file() and graph.is_file() and structure["pass"]),
        "eager_exact": exact,
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "cache_shape": list(CACHE_SHAPE),
        "cache_bytes_per_tensor": 2 * torch.tensor(CACHE_SHAPE).prod().item(),
        "slots": list(SLOTS),
        "structure_pass": structure["pass"],
        "claim_boundary": (
            "Weight-free single-layer eager semantics and AIR structure only; "
            "no GraphPp, shared-lease, full-model, or performance claim."
        ),
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
