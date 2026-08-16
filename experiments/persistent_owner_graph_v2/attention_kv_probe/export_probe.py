#!/usr/bin/env python3
"""Export the FunctionPp-owned PA-NZ update-to-FIA safety graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch_npu
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.npu_export import dynamo_export
from vllm_ascend.utils import enable_custom_op


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Reuse the already-audited explicit 31-slot public FIA converter.
from experiments.persistent_decoder_p3.fia_graphpp_probe.export_mini_fia import (  # noqa: E402,F401
    FIA_INPUT_NAMES,
)

from inspect_probe_graph import inspect_graph  # noqa: E402


BATCH = 4
QUERY_HEADS = 28
KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 128
BLOCKS_PER_ROW = 3
PHYSICAL_BLOCKS = BATCH * BLOCKS_PER_ROW
PACK_WIDTH = 16
PACKS_PER_HEAD = HEAD_DIM // PACK_WIDTH
KV_SHAPE = (PHYSICAL_BLOCKS, KV_HEADS, PACKS_PER_HEAD, BLOCK_SIZE, PACK_WIDTH)
QUERY_SHAPE = (BATCH, QUERY_HEADS, 1, HEAD_DIM)
MASK_SHAPE = (BATCH, 1, 1, BLOCK_SIZE * BLOCKS_PER_ROW)
METADATA_ELEMENTS = 5
TICKET_ELEMENTS = 9
POSITIONS = (0, 127, 128, 383)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


enable_custom_op()
_lib = torch.library.Library("cruise_device_paged_kv_attention", "DEF")
_lib.define(
    "update(Tensor key_cache, Tensor value_cache, Tensor metadata) -> Tensor"
)
_lib.define("order_query(Tensor query, Tensor ticket) -> Tensor")


def _update_npu(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: torch.Tensor,
) -> torch.Tensor:
    del key_cache, value_cache, metadata
    return torch.empty((TICKET_ELEMENTS,), dtype=torch.int32, device="npu")


def _update_meta(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: torch.Tensor,
) -> torch.Tensor:
    del key_cache, value_cache, metadata
    return torch.empty((TICKET_ELEMENTS,), dtype=torch.int32, device="meta")


def _order_npu(query: torch.Tensor, ticket: torch.Tensor) -> torch.Tensor:
    del ticket
    return torch.empty_like(query)


def _order_meta(query: torch.Tensor, ticket: torch.Tensor) -> torch.Tensor:
    del ticket
    return torch.empty_like(query, device="meta")


_lib.impl("update", _update_npu, "PrivateUse1")
_lib.impl("update", _update_meta, "Meta")
_lib.impl("order_query", _order_npu, "PrivateUse1")
_lib.impl("order_query", _order_meta, "Meta")


@register_fx_node_ge_converter(
    torch.ops.cruise_device_paged_kv_attention.update.default
)
def convert_update(
    key_cache: Tensor,
    value_cache: Tensor,
    metadata: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    del meta_outputs
    return torchair.ge.custom_op(
        "DevicePagedKvUpdate",
        inputs={
            "key_cache": key_cache,
            "value_cache": value_cache,
            "metadata": metadata,
        },
        outputs=["ticket"],
    )


@register_fx_node_ge_converter(
    torch.ops.cruise_device_paged_kv_attention.order_query.default
)
def convert_order_query(
    query: Tensor,
    ticket: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    del meta_outputs
    return torchair.ge.custom_op(
        "DeviceQueryAfterKvUpdate",
        inputs={"query": query, "ticket": ticket},
        outputs=["ordered_query"],
    )


class AttentionKvProbe(torch.nn.Module):
    def forward(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ticket = torch.ops.cruise_device_paged_kv_attention.update(
            key_cache, value_cache, metadata
        )
        ordered_query = torch.ops.cruise_device_paged_kv_attention.order_query(
            query, ticket
        )
        attention = torch_npu.npu_fused_infer_attention_score(
            ordered_query,
            key_cache,
            value_cache,
            num_heads=QUERY_HEADS,
            num_key_value_heads=KV_HEADS,
            input_layout="BNSD",
            atten_mask=mask,
            actual_seq_lengths_kv=[BLOCK_SIZE * BLOCKS_PER_ROW] * BATCH,
            block_table=block_table,
            block_size=BLOCK_SIZE,
            scale=HEAD_DIM**-0.5,
            sparse_mode=0,
        )[0]
        return attention, ticket


def inputs() -> tuple[torch.Tensor, ...]:
    key_cache = torch.zeros(KV_SHAPE, dtype=torch.bfloat16, device="npu")
    value_cache = torch.zeros_like(key_cache)
    query = torch.ones(QUERY_SHAPE, dtype=torch.bfloat16, device="npu")
    metadata = torch.tensor((1, *POSITIONS), dtype=torch.int32, device="npu")
    mask = torch.ones(MASK_SHAPE, dtype=torch.bool, device="npu")
    for row, position in enumerate(POSITIONS):
        mask[row, 0, 0, position] = False
    block_table = torch.arange(
        PHYSICAL_BLOCKS, dtype=torch.int32, device="npu"
    ).reshape(BATCH, BLOCKS_PER_ROW)
    return key_cache, value_cache, metadata, query, mask, block_table


def inspect_air(path: Path) -> dict[str, object]:
    from torchair._ge_concrete_graph.ge_ir_pb2 import ModelDef

    try:
        model = ModelDef()
        model.ParseFromString(path.read_bytes())
        nodes = [op for graph in model.graph for op in graph.op]
        counts: dict[str, int] = {}
        for node in nodes:
            counts[node.type] = counts.get(node.type, 0) + 1
        fia = [node for node in nodes if node.type == "FusedInferAttentionScore"]
        return {
            "air_op_counts": dict(sorted(counts.items())),
            "fia_node_count": len(fia),
            "fia_slot_contract_pass": (
                len(fia) == 1
                and len(fia[0].input) == len(FIA_INPUT_NAMES)
                and len(fia[0].input_desc) == len(FIA_INPUT_NAMES)
            ),
        }
    except Exception as error:
        return {"fia_slot_contract_pass": False, "air_error": repr(error)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    torch.npu.set_device(0)
    sample_inputs = inputs()
    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        *sample_inputs,
        model=AttentionKvProbe().eval().npu(),
        export_path=str(args.output_dir),
        export_name="attention_kv_probe",
        dynamic=False,
        config=config,
    )
    torch.npu.synchronize()

    air = args.output_dir / "attention_kv_probe.air"
    graph = args.output_dir / "dynamo.pbtxt"
    structure = inspect_graph(graph) if graph.is_file() else {"pass": False}
    air_result = inspect_air(air) if air.is_file() else {
        "fia_slot_contract_pass": False
    }
    (args.output_dir / "graph-structure.json").write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    abi = {
        "schema_version": 1,
        "gate": "V2-KV-ATTENTION-ABI",
        "pass": structure.get("data_input_abi_pass") is True,
        "graph_sha256": structure.get("graph_sha256"),
        "data_inputs": structure.get("data_input_abi", []),
    }
    (args.output_dir / "graph-abi.json").write_text(
        json.dumps(abi, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "gate": "V2-KV-ATTENTION-EXPORT",
        "pass": bool(
            air.is_file()
            and graph.is_file()
            and structure.get("pass") is True
            and abi["pass"] is True
            and air_result.get("fia_slot_contract_pass") is True
        ),
        "kv_shape": list(KV_SHAPE),
        "kv_bytes": 2 * torch.tensor(KV_SHAPE).prod().item() * 2,
        "metadata_bytes": METADATA_ELEMENTS * 4,
        "attention_output_bytes": torch.tensor(QUERY_SHAPE).prod().item() * 2,
        "ticket_output_bytes": TICKET_ELEMENTS * 4,
        "positions": list(POSITIONS),
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "structure_pass": structure.get("pass") is True,
        "data_input_abi_pass": abi["pass"],
        "claim_boundary": (
            "Combined PA-NZ update-to-FIA export only; no execution, full "
            "Decoder, zero-copy, or P5 qualification claim."
        ),
        **air_result,
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("V2_KV_ATTENTION_EXPORT " + json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
