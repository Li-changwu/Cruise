#!/usr/bin/env python3
"""Render P2 FunctionPp, GraphPp, toolchain, and deployment configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_function_config(workspace: Path, toolchain_config: Path) -> dict:
    workspace = workspace.resolve(strict=True)
    toolchain_config = toolchain_config.resolve(strict=True)
    if not (workspace / "CMakeLists.txt").is_file():
        raise ValueError("controller workspace has no CMakeLists.txt")
    if not (workspace / "persistent_cancel_p2.cpp").is_file():
        raise ValueError("controller workspace has no P2 source")
    return {
        "func_list": [
            {
                "func_name": "persistent_cancel_p2",
                "inputs_index": [0],
                "outputs_index": [0],
                "stream_input": True,
            }
        ],
        "input_num": 1,
        "output_num": 1,
        "target_bin": "libpersistent_cancel_p2.so",
        "workspace": str(workspace),
        "cmakelist_path": "CMakeLists.txt",
        "heavy_load": False,
        "compiler": str(toolchain_config),
    }


def make_graph_config() -> dict:
    return {
        "build_options": {"ge.modelFileNamePrefix": "persistent_cancel_p2_add"},
        "inputs_tensor_desc": [
            {"data_type": "DT_INT32", "shape": [4]},
            {"data_type": "DT_INT32", "shape": [4]},
        ],
    }


def make_toolchain_config(ascend_toolchain: Path) -> dict:
    ascend_toolchain = ascend_toolchain.resolve(strict=True)
    if not ascend_toolchain.is_file():
        raise ValueError("Ascend FunctionPp toolchain is not a file")
    return {
        "compiler": [
            {"resource_type": "Ascend", "toolchain": str(ascend_toolchain)}
        ]
    }


def make_deploy_config() -> dict:
    return {
        "batch_deploy_info": [
            {
                "flow_node_list": ["persistent_cancel_p2_node"],
                "logic_device_list": "0:0:0:0",
            }
        ]
    }


def _write(path: Path, value: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ascend-toolchain", type=Path, required=True)
    parser.add_argument("--function-output", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--toolchain-output", type=Path, required=True)
    parser.add_argument("--deploy-output", type=Path, required=True)
    args = parser.parse_args()
    _write(args.toolchain_output, make_toolchain_config(args.ascend_toolchain))
    _write(
        args.function_output,
        make_function_config(args.workspace, args.toolchain_output),
    )
    _write(args.graph_output, make_graph_config())
    _write(args.deploy_output, make_deploy_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
