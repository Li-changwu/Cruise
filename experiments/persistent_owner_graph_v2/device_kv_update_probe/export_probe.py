#!/usr/bin/env python3
"""Export the minimal Device-owned KV slot update graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.npu_export import dynamo_export
from vllm_ascend.utils import enable_custom_op

from inspect_probe_graph import inspect_graph


CACHE_ELEMENTS = 4096
REPORT_ELEMENTS = 9
SLOT = 2048


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


enable_custom_op()
_lib = torch.library.Library("cruise_device_kv_update", "DEF")
_lib.define("update(Tensor cache, Tensor metadata) -> Tensor")


def _update_npu(cache: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
    del metadata
    return torch.empty((REPORT_ELEMENTS,), dtype=torch.int32, device=cache.device)


def _update_meta(cache: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
    del cache, metadata
    return torch.empty((REPORT_ELEMENTS,), dtype=torch.int32, device="meta")


_lib.impl("update", _update_npu, "PrivateUse1")
_lib.impl("update", _update_meta, "Meta")


@register_fx_node_ge_converter(torch.ops.cruise_device_kv_update.update.default)
def convert_update(
    cache: Tensor, metadata: Tensor, *, meta_outputs: TensorSpec = None
):
    del meta_outputs
    return torchair.ge.custom_op(
        "DeviceKvSlotUpdate",
        inputs={"cache": cache, "metadata": metadata},
        outputs=["report"],
    )


class DeviceKvUpdateProbe(torch.nn.Module):
    def forward(self, cache: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return torch.ops.cruise_device_kv_update.update(cache, metadata)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.npu.set_device(0)
    cache = torch.zeros((CACHE_ELEMENTS,), dtype=torch.bfloat16, device="npu")
    metadata = torch.tensor([1, SLOT, 0, 0x3F80], dtype=torch.int32, device="npu")
    model = DeviceKvUpdateProbe().eval().npu()
    with torch.no_grad():
        eager_report = model(cache, metadata)
    torch.npu.synchronize()
    eager_dispatch = eager_report.shape == (REPORT_ELEMENTS,)

    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        cache,
        metadata,
        model=model,
        export_path=str(args.output_dir),
        export_name="device_kv_update_probe",
        dynamic=False,
        config=config,
    )
    torch.npu.synchronize()
    air = args.output_dir / "device_kv_update_probe.air"
    graph = args.output_dir / "dynamo.pbtxt"
    structure = (
        inspect_graph(graph)
        if graph.is_file()
        else {"pass": False, "error": "dynamo.pbtxt was not exported"}
    )
    (args.output_dir / "graph-structure.json").write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "gate": "V2-DEVICE-KV-UPDATE-EXPORT",
        "pass": bool(
            eager_dispatch and air.is_file() and graph.is_file() and structure["pass"]
        ),
        "eager_dispatch": eager_dispatch,
        "cache_elements": CACHE_ELEMENTS,
        "probe_cache_bytes": CACHE_ELEMENTS * 2,
        "report_elements": REPORT_ELEMENTS,
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "structure_pass": structure["pass"],
        "claim_boundary": (
            "Synthetic state-buffer export and structure only; no target KV, "
            "full Decoder, zero-copy, or P5 claim."
        ),
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("V2_DEVICE_KV_UPDATE_EXPORT " + json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
