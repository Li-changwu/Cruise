#!/usr/bin/env python3
"""Export the B4/K384 Device Paged-KV update/read ordering graph."""

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


BATCH_SIZE = 4
BLOCKS_PER_ROW = 3
PHYSICAL_BLOCKS = BATCH_SIZE * BLOCKS_PER_ROW
PACKED_CHANNELS = 32
BLOCK_SIZE = 128
PACK_WIDTH = 16
STATE_SHAPE = (2, PHYSICAL_BLOCKS, PACKED_CHANNELS, BLOCK_SIZE, PACK_WIDTH)
METADATA_ELEMENTS = 5
TICKET_ELEMENTS = 9
REPORT_ELEMENTS = 18
POSITIONS = (0, 127, 128, 383)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


enable_custom_op()
_lib = torch.library.Library("cruise_device_paged_kv", "DEF")
_lib.define("update(Tensor state, Tensor metadata) -> Tensor")
_lib.define("read(Tensor state, Tensor metadata, Tensor ticket) -> Tensor")


def _update_npu(state: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
    del state, metadata
    return torch.empty((TICKET_ELEMENTS,), dtype=torch.int32, device="npu")


def _update_meta(state: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
    del state, metadata
    return torch.empty((TICKET_ELEMENTS,), dtype=torch.int32, device="meta")


def _read_npu(
    state: torch.Tensor, metadata: torch.Tensor, ticket: torch.Tensor
) -> torch.Tensor:
    del state, metadata, ticket
    return torch.empty((REPORT_ELEMENTS,), dtype=torch.int32, device="npu")


def _read_meta(
    state: torch.Tensor, metadata: torch.Tensor, ticket: torch.Tensor
) -> torch.Tensor:
    del state, metadata, ticket
    return torch.empty((REPORT_ELEMENTS,), dtype=torch.int32, device="meta")


_lib.impl("update", _update_npu, "PrivateUse1")
_lib.impl("update", _update_meta, "Meta")
_lib.impl("read", _read_npu, "PrivateUse1")
_lib.impl("read", _read_meta, "Meta")


@register_fx_node_ge_converter(torch.ops.cruise_device_paged_kv.update.default)
def convert_update(
    state: Tensor, metadata: Tensor, *, meta_outputs: TensorSpec = None
):
    del meta_outputs
    return torchair.ge.custom_op(
        "DevicePagedKvUpdate",
        inputs={"state": state, "metadata": metadata},
        outputs=["ticket"],
    )


@register_fx_node_ge_converter(torch.ops.cruise_device_paged_kv.read.default)
def convert_read(
    state: Tensor,
    metadata: Tensor,
    ticket: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    del meta_outputs
    return torchair.ge.custom_op(
        "DevicePagedKvRead",
        inputs={"state": state, "metadata": metadata, "ticket": ticket},
        outputs=["report"],
    )


class PagedKvOrderProbe(torch.nn.Module):
    def forward(self, state: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        ticket = torch.ops.cruise_device_paged_kv.update(state, metadata)
        return torch.ops.cruise_device_paged_kv.read(state, metadata, ticket)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.npu.set_device(0)
    state = torch.zeros(STATE_SHAPE, dtype=torch.bfloat16, device="npu")
    metadata = torch.tensor((1, *POSITIONS), dtype=torch.int32, device="npu")
    model = PagedKvOrderProbe().eval().npu()
    with torch.no_grad():
        eager_report = model(state, metadata)
    torch.npu.synchronize()
    eager_dispatch = eager_report.shape == (REPORT_ELEMENTS,)

    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        state,
        metadata,
        model=model,
        export_path=str(args.output_dir),
        export_name="paged_kv_order_probe",
        dynamic=False,
        config=config,
    )
    torch.npu.synchronize()
    air = args.output_dir / "paged_kv_order_probe.air"
    graph = args.output_dir / "dynamo.pbtxt"
    structure = (
        inspect_graph(graph)
        if graph.is_file()
        else {"pass": False, "error": "dynamo.pbtxt was not exported"}
    )
    (args.output_dir / "graph-structure.json").write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state_elements = torch.tensor(STATE_SHAPE).prod().item()
    result = {
        "gate": "V2-KV-ORDER-EXPORT",
        "pass": bool(
            eager_dispatch and air.is_file() and graph.is_file() and structure["pass"]
        ),
        "eager_dispatch": eager_dispatch,
        "state_shape": list(STATE_SHAPE),
        "state_bytes": state_elements * 2,
        "metadata_elements": METADATA_ELEMENTS,
        "ticket_elements": TICKET_ELEMENTS,
        "report_elements": REPORT_ELEMENTS,
        "positions": list(POSITIONS),
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "structure_pass": structure["pass"],
        "claim_boundary": (
            "B4/K384 Paged-KV custom-op export and dependency structure only; "
            "no execution, attention, full Decoder, zero-copy, or P5 claim."
        ),
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("V2_KV_ORDER_EXPORT " + json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
