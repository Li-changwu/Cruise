#!/usr/bin/env python3
"""Render GraphPp, FunctionPp, and deployment configs for the combined probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KV_SHAPE = [12, 4, 8, 128, 16]


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ascend-toolchain", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--function-output", type=Path, required=True)
    parser.add_argument("--toolchain-output", type=Path, required=True)
    parser.add_argument("--deploy-output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    toolchain = args.ascend_toolchain.resolve(strict=True)
    write(
        args.graph_output,
        {
            "build_options": {"ge.modelFileNamePrefix": "cruise_v2_attention_kv_probe"},
            "inputs_tensor_desc": [
                {"data_type": "DT_BFLOAT16", "shape": KV_SHAPE},
                {"data_type": "DT_BFLOAT16", "shape": KV_SHAPE},
                {"data_type": "DT_BFLOAT16", "shape": [4, 28, 1, 128]},
                {"data_type": "DT_INT32", "shape": [5]},
                {"data_type": "DT_BOOL", "shape": [4, 1, 1, 384]},
                {"data_type": "DT_INT32", "shape": [4, 3]},
            ],
        },
    )
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
