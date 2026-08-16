#!/usr/bin/env python3
"""Render GraphPp, FunctionPp, and deployment configs for the combined probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_ROLES = (
    "key_cache",
    "value_cache",
    "metadata",
    "query",
    "mask",
    "block_table",
)
EXPECTED_CONTRACTS = {
    "key_cache": ("DT_BF16", [12, 4, 8, 128, 16]),
    "value_cache": ("DT_BF16", [12, 4, 8, 128, 16]),
    "metadata": ("DT_INT32", [5]),
    "query": ("DT_BF16", [4, 28, 1, 128]),
    "mask": ("DT_BOOL", [4, 1, 1, 384]),
    "block_table": ("DT_INT32", [4, 3]),
}
GRAPH_CONFIG_DTYPES = {
    "DT_BF16": "DT_BFLOAT16",
    "DT_INT32": "DT_INT32",
    "DT_BOOL": "DT_BOOL",
}
CONSTANT_NAMES = {
    "key_cache": "kKeyCacheInput",
    "value_cache": "kValueCacheInput",
    "metadata": "kMetadataInput",
    "query": "kQueryInput",
    "mask": "kMaskInput",
    "block_table": "kBlockTableInput",
}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_abi(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    inputs = sorted(payload.get("data_inputs", []), key=lambda item: item["index"])
    if (
        payload.get("schema_version") != 1
        or payload.get("pass") is not True
        or len(inputs) != len(EXPECTED_ROLES)
        or [item.get("index") for item in inputs] != list(range(len(inputs)))
        or tuple(item.get("role") for item in inputs) != EXPECTED_ROLES
    ):
        raise ValueError("AIR-derived attention KV ABI is incomplete or reordered")
    for item in inputs:
        expected_dtype, expected_shape = EXPECTED_CONTRACTS[item["role"]]
        if (
            item.get("data_type") != expected_dtype
            or item.get("shape") != expected_shape
        ):
            raise ValueError("AIR-derived attention KV ABI type or shape mismatch")
    return inputs


def graph_config(inputs: list[dict]) -> dict:
    return {
        "build_options": {"ge.modelFileNamePrefix": "cruise_v2_attention_kv_probe"},
        "inputs_tensor_desc": [
            {
                "data_type": GRAPH_CONFIG_DTYPES[item["data_type"]],
                "shape": item["shape"],
            }
            for item in inputs
        ],
    }


def render_header(inputs: list[dict]) -> str:
    indices = {item["role"]: item["index"] for item in inputs}
    lines = [
        "#ifndef CRUISE_ATTENTION_KV_GRAPH_ABI_H_",
        "#define CRUISE_ATTENTION_KV_GRAPH_ABI_H_",
        "",
        "#include <cstddef>",
        "",
        "namespace CruiseAttentionKvGraphAbi {",
        f"constexpr std::size_t kInputCount = {len(inputs)}U;",
    ]
    lines.extend(
        f"constexpr std::size_t {CONSTANT_NAMES[role]} = {indices[role]}U;"
        for role in EXPECTED_ROLES
    )
    lines.extend(
        [
            "}  // namespace CruiseAttentionKvGraphAbi",
            "",
            "#endif  // CRUISE_ATTENTION_KV_GRAPH_ABI_H_",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--abi-input", type=Path, required=True)
    parser.add_argument("--header-output", type=Path, required=True)
    parser.add_argument("--ascend-toolchain", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--function-output", type=Path, required=True)
    parser.add_argument("--toolchain-output", type=Path, required=True)
    parser.add_argument("--deploy-output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    toolchain = args.ascend_toolchain.resolve(strict=True)
    inputs = load_abi(args.abi_input.resolve(strict=True))
    write(args.graph_output, graph_config(inputs))
    args.header_output.parent.mkdir(parents=True, exist_ok=True)
    args.header_output.write_text(render_header(inputs), encoding="utf-8")
    write(
        args.function_output,
        {
            "func_list": [
                {
                    "func_name": "attention_kv_controller",
                    "inputs_index": [0],
                    "outputs_index": [0],
                }
            ],
            "input_num": 1,
            "output_num": 1,
            "target_bin": "libattention_kv_controller.so",
            "workspace": str(workspace),
            "cmakelist_path": "CMakeLists.txt",
            "heavy_load": False,
            "compiler": str(args.toolchain_output.resolve()),
        },
    )
    write(
        args.toolchain_output,
        {"compiler": [{"resource_type": "Ascend", "toolchain": str(toolchain)}]},
    )
    write(
        args.deploy_output,
        {
            "batch_deploy_info": [
                {
                    "flow_node_list": ["attention_kv_controller_node"],
                    "logic_device_list": "0:0:0:0",
                }
            ]
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
