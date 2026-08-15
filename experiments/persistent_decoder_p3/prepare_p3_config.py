#!/usr/bin/env python3
"""Render P3 FunctionPp, imported AIR GraphPp, toolchain, and deployment configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_function_config(workspace: Path, toolchain_config: Path) -> dict:
    workspace = workspace.resolve(strict=True)
    toolchain_config = toolchain_config.resolve(strict=True)
    if not (workspace / "CMakeLists.txt").is_file():
        raise ValueError("controller workspace has no CMakeLists.txt")
    if not (workspace / "persistent_decoder_p3.cpp").is_file():
        raise ValueError("controller workspace has no P3 source")
    return {
        "func_list": [
            {
                "func_name": "persistent_decoder_p3",
                "inputs_index": [0],
                "outputs_index": [0],
                "stream_input": True,
            }
        ],
        "input_num": 1,
        "output_num": 1,
        "target_bin": "libpersistent_decoder_p3.so",
        "workspace": str(workspace),
        "cmakelist_path": "CMakeLists.txt",
        "heavy_load": True,
        "compiler": str(toolchain_config),
    }


def make_graph_config() -> dict:
    cache = [28, 12, 128, 4, 128]
    return {
        "build_options": {
            "ge.externalWeight": "1",
            "ge.modelFileNamePrefix": "persistent_decoder_p3_step",
        },
        "inputs_tensor_desc": [
            {"data_type": "DT_INT64", "shape": [4, 1]},
            {"data_type": "DT_INT64", "shape": [4]},
            {"data_type": "DT_INT32", "shape": [4, 1]},
            {"data_type": "DT_BFLOAT16", "shape": cache},
            {"data_type": "DT_INT32", "shape": [4]},
            {"data_type": "DT_INT32", "shape": [4]},
            {"data_type": "DT_INT32", "shape": [4, 3]},
            {"data_type": "DT_BFLOAT16", "shape": cache},
        ],
    }


def make_toolchain_config(ascend_toolchain: Path) -> dict:
    ascend_toolchain = ascend_toolchain.resolve(strict=True)
    if not ascend_toolchain.is_file():
        raise ValueError("Ascend FunctionPp toolchain is not a file")
    return {"compiler": [{"resource_type": "Ascend", "toolchain": str(ascend_toolchain)}]}


def make_deploy_config() -> dict:
    return {
        "batch_deploy_info": [
            {
                "flow_node_list": ["persistent_decoder_p3_node"],
                "logic_device_list": "0:0:0:0",
            }
        ]
    }


def write(path: Path, value: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ascend-toolchain", type=Path, required=True)
    parser.add_argument("--function-output", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--toolchain-output", type=Path, required=True)
    parser.add_argument("--deploy-output", type=Path, required=True)
    args = parser.parse_args()
    write(args.toolchain_output, make_toolchain_config(args.ascend_toolchain))
    write(args.function_output, make_function_config(args.workspace, args.toolchain_output))
    write(args.graph_output, make_graph_config())
    write(args.deploy_output, make_deploy_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
