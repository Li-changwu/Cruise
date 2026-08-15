#!/usr/bin/env python3
"""Render GraphPp, FunctionPp, toolchain, and deployment configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def graph_config() -> dict:
    return {
        "build_options": {
            "ge.modelFileNamePrefix": "cruise_v2_device_kv_update_probe"
        },
        "inputs_tensor_desc": [
            {"data_type": "DT_BFLOAT16", "shape": [4096]},
            {"data_type": "DT_INT32", "shape": [4]},
        ],
    }


def function_config(workspace: Path, toolchain: Path) -> dict:
    workspace = workspace.resolve(strict=True)
    toolchain = toolchain.resolve(strict=True)
    if not (workspace / "CMakeLists.txt").is_file():
        raise ValueError("Device KV update controller has no CMakeLists.txt")
    if not (workspace / "device_kv_update_controller.cpp").is_file():
        raise ValueError("Device KV update controller source is missing")
    return {
        "func_list": [
            {
                "func_name": "device_kv_update_controller",
                "inputs_index": [0],
                "outputs_index": [0],
            }
        ],
        "input_num": 1,
        "output_num": 1,
        "target_bin": "libdevice_kv_update_controller.so",
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
                "flow_node_list": ["device_kv_update_controller_node"],
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
    write(
        args.function_output,
        function_config(args.workspace, args.toolchain_output),
    )
    write(args.deploy_output, deploy_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
