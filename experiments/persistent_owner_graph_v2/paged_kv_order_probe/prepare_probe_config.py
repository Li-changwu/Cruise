#!/usr/bin/env python3
"""Render GraphPp, FunctionPp, toolchain, and deployment configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STATE_SHAPE = [2, 12, 32, 128, 16]


def graph_config() -> dict:
    return {
        "build_options": {"ge.modelFileNamePrefix": "cruise_v2_paged_kv_order_probe"},
        "inputs_tensor_desc": [
            {"data_type": "DT_BFLOAT16", "shape": STATE_SHAPE},
            {"data_type": "DT_INT32", "shape": [5]},
        ],
    }


def function_config(workspace: Path, toolchain: Path) -> dict:
    workspace = workspace.resolve(strict=True)
    toolchain = toolchain.resolve(strict=True)
    if not (workspace / "CMakeLists.txt").is_file():
        raise ValueError("Paged-KV order controller has no CMakeLists.txt")
    if not (workspace / "paged_kv_order_controller.cpp").is_file():
        raise ValueError("Paged-KV order controller source is missing")
    return {
        "func_list": [
            {
                "func_name": "paged_kv_order_controller",
                "inputs_index": [0],
                "outputs_index": [0],
            }
        ],
        "input_num": 1,
        "output_num": 1,
        "target_bin": "libpaged_kv_order_controller.so",
        "workspace": str(workspace),
        "cmakelist_path": "CMakeLists.txt",
        "heavy_load": False,
        "compiler": str(toolchain),
    }


def toolchain_config(ascend_toolchain: Path) -> dict:
    ascend_toolchain = ascend_toolchain.resolve(strict=True)
    if not ascend_toolchain.is_file():
        raise ValueError("Ascend FunctionPp toolchain is not a file")
    return {
        "compiler": [
            {"resource_type": "Ascend", "toolchain": str(ascend_toolchain)}
        ]
    }


def deploy_config() -> dict:
    return {
        "batch_deploy_info": [
            {
                "flow_node_list": ["paged_kv_order_controller_node"],
                "logic_device_list": "0:0:0:0",
            }
        ]
    }


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
    write(args.toolchain_output, toolchain_config(args.ascend_toolchain))
    write(args.graph_output, graph_config())
    write(args.function_output, function_config(args.workspace, args.toolchain_output))
    write(args.deploy_output, deploy_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
