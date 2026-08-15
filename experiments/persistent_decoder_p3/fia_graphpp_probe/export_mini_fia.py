#!/usr/bin/env python3
"""Export a weight-free single-FIA graph for GraphPp packaging diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import torch
import torch_npu
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import (
    register_fx_node_ge_converter,
)
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, TensorSpec

# Load the built-in converter set before installing this probe-local override.
from torchair._ge_concrete_graph.ge_converter import custom as _custom_converters  # noqa: F401


BATCH = 4
QUERY_HEADS = 28
KV_HEADS = 4
QUERY_TOKENS = 1
KV_TOKENS = 384
HEAD_DIM = 128

FIA_INPUT_NAMES = (
    "query",
    "key0",
    "value0",
    "pse_shift",
    "atten_mask",
    "actual_seq_lengths",
    "actual_seq_lengths_kv",
    "dequant_scale1",
    "quant_scale1",
    "dequant_scale2",
    "quant_scale2",
    "quant_offset2",
    "antiquant_scale",
    "antiquant_offset",
    "block_table",
    "query_padding_size",
    "kv_padding_size",
    "key_antiquant_scale",
    "key_antiquant_offset",
    "value_antiquant_scale",
    "value_antiquant_offset",
    "key_shared_prefix",
    "value_shared_prefix",
    "actual_shared_prefix_len",
    "query_rope",
    "key_rope",
    "key_rope_antiquant_scale",
    "dequant_scale_query",
    "learnable_sink",
    "q_start_idx",
    "kv_start_idx",
)


@register_fx_node_ge_converter(
    torch.ops.npu.npu_fused_infer_attention_score.default
)
def convert_mini_fia_with_explicit_optional_slots(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    pse_shift: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    actual_seq_lengths: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    dequant_scale1: Optional[Tensor] = None,
    quant_scale1: Optional[Tensor] = None,
    dequant_scale2: Optional[Tensor] = None,
    quant_scale2: Optional[Tensor] = None,
    quant_offset2: Optional[Tensor] = None,
    antiquant_scale: Optional[Tensor] = None,
    antiquant_offset: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    query_padding_size: Optional[Tensor] = None,
    kv_padding_size: Optional[Tensor] = None,
    key_antiquant_scale: Optional[Tensor] = None,
    key_antiquant_offset: Optional[Tensor] = None,
    value_antiquant_scale: Optional[Tensor] = None,
    value_antiquant_offset: Optional[Tensor] = None,
    key_shared_prefix: Optional[Tensor] = None,
    value_shared_prefix: Optional[Tensor] = None,
    actual_shared_prefix_len: Optional[Tensor] = None,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
    key_rope_antiquant_scale: Optional[Tensor] = None,
    num_heads: int = 1,
    scale: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: str = "BSH",
    num_key_value_heads: int = 0,
    sparse_mode: int = 0,
    inner_precise: int = 0,
    block_size: int = 0,
    antiquant_mode: int = 0,
    softmax_lse_flag: bool = False,
    key_antiquant_mode: int = 0,
    value_antiquant_mode: int = 0,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "FusedInferAttentionScore",
        inputs={
            "query": query,
            "key": [key],
            "value": [value],
            "pse_shift": pse_shift,
            "atten_mask": atten_mask,
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "dequant_scale1": dequant_scale1,
            "quant_scale1": quant_scale1,
            "dequant_scale2": dequant_scale2,
            "quant_scale2": quant_scale2,
            "quant_offset2": quant_offset2,
            "antiquant_scale": antiquant_scale,
            "antiquant_offset": antiquant_offset,
            "block_table": block_table,
            "query_padding_size": query_padding_size,
            "kv_padding_size": kv_padding_size,
            "key_antiquant_scale": key_antiquant_scale,
            "key_antiquant_offset": key_antiquant_offset,
            "value_antiquant_scale": value_antiquant_scale,
            "value_antiquant_offset": value_antiquant_offset,
            "key_shared_prefix": key_shared_prefix,
            "value_shared_prefix": value_shared_prefix,
            "actual_shared_prefix_len": actual_shared_prefix_len,
            "query_rope": query_rope,
            "key_rope": key_rope,
            "key_rope_antiquant_scale": key_rope_antiquant_scale,
            "dequant_scale_query": None,
            "learnable_sink": None,
            "q_start_idx": None,
            "kv_start_idx": None,
        },
        attrs={
            "num_heads": attr.Int(num_heads),
            "scale": attr.Float(scale),
            "pre_tokens": attr.Int(pre_tokens),
            "next_tokens": attr.Int(next_tokens),
            "input_layout": attr.Str(input_layout),
            "num_key_value_heads": attr.Int(num_key_value_heads),
            "sparse_mode": attr.Int(sparse_mode),
            "inner_precise": attr.Int(inner_precise),
            "block_size": attr.Int(block_size),
            "antiquant_mode": attr.Int(antiquant_mode),
            "softmax_lse_flag": attr.Bool(softmax_lse_flag),
            "key_antiquant_mode": attr.Int(key_antiquant_mode),
            "value_antiquant_mode": attr.Int(value_antiquant_mode),
            "query_quant_mode": attr.Int(0),
            "pse_type": attr.Int(0),
            "out_dtype": attr.Int(0),
        },
        outputs=["attention_out", "softmax_lse"],
    )


class MiniFIA(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        attention = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=QUERY_HEADS,
            num_key_value_heads=KV_HEADS,
            input_layout="BNSD",
            atten_mask=mask,
            scale=HEAD_DIM**-0.5,
            sparse_mode=0,
        )[0]
        return attention.transpose(1, 2).reshape(
            BATCH, QUERY_TOKENS, QUERY_HEADS * HEAD_DIM
        )


def inputs() -> tuple[torch.Tensor, ...]:
    query = torch.ones(
        (BATCH, QUERY_HEADS, QUERY_TOKENS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="npu",
    )
    key = torch.ones(
        (BATCH, KV_HEADS, KV_TOKENS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="npu",
    )
    value = torch.ones_like(key)
    mask = torch.zeros(
        (BATCH, 1, QUERY_TOKENS, KV_TOKENS),
        dtype=torch.bool,
        device="npu",
    )
    return query, key, value, mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_fia_air(path: Path) -> dict[str, object]:
    from torchair._ge_concrete_graph.ge_ir_pb2 import ModelDef

    try:
        model = ModelDef()
        model.ParseFromString(path.read_bytes())
        nodes = [
            op
            for graph in model.graph
            for op in graph.op
            if op.type == "FusedInferAttentionScore"
        ]
        if len(nodes) != 1:
            return {
                "fia_node_count": len(nodes),
                "fia_slot_contract_pass": False,
            }
        node = nodes[0]
        input_names = [desc.name for desc in node.input_desc]
        attr_names = [
            value.decode("utf-8")
            for value in node.attr["_input_name_key"].list.s
        ]
        attr_values = list(node.attr["_input_name_value"].list.i)
        expected_names = list(FIA_INPUT_NAMES)
        slot_names = [None] * len(expected_names)
        attr_mapping_valid = (
            len(attr_names) == len(expected_names)
            and sorted(attr_values) == list(range(len(expected_names)))
        )
        if attr_mapping_valid:
            for name, slot in zip(attr_names, attr_values):
                slot_names[slot] = name
        return {
            "fia_node_count": 1,
            "fia_input_count": len(node.input),
            "fia_input_desc_count": len(node.input_desc),
            "fia_input_names": input_names,
            "fia_input_name_attr": attr_names,
            "fia_input_value_attr": attr_values,
            "fia_slot_names": slot_names,
            "fia_slot_contract_pass": (
                len(node.input) == len(expected_names)
                and len(node.input_desc) == len(expected_names)
                and attr_mapping_valid
                and slot_names == expected_names
            ),
        }
    except Exception as error:
        return {
            "fia_slot_contract_pass": False,
            "fia_air_inspection_error": repr(error),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    torch.npu.set_device(0)
    model = MiniFIA().eval().npu()
    sample_inputs = inputs()

    from torchair.configs.compiler_config import CompilerConfig
    from torchair.npu_export import dynamo_export

    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        *sample_inputs,
        model=model,
        export_path=str(args.output_dir),
        export_name="mini_fia",
        dynamic=False,
        config=config,
    )
    air = args.output_dir / "mini_fia.air"
    graph = args.output_dir / "dynamo.pbtxt"
    fia_air = inspect_fia_air(air) if air.is_file() else {
        "fia_slot_contract_pass": False
    }
    result = {
        "gate": "P3-MINI-FIA-EXPORT",
        "pass": (
            air.is_file()
            and graph.is_file()
            and fia_air["fia_slot_contract_pass"] is True
        ),
        "eager_exact": None,
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "external_file_count": len(
            [path for path in args.output_dir.iterdir() if path.is_file() and not path.suffix]
        ),
        "claim_boundary": (
            "Weight-free single FusedInferAttentionScore export only; execution "
            "is checked in a separate Graph process, with no P3 Owner claim."
        ),
        **fia_air,
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
