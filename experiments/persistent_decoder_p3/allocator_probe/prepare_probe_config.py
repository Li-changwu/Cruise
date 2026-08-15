#!/usr/bin/env python3
"""Render FunctionPp and deployment configs for the P3 allocator probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def function_config(
    workspace: Path, toolchain_config: Path, heavy_load: bool
) -> dict:
    workspace = workspace.resolve(strict=True)
    toolchain_config = toolchain_config.resolve(strict=True)
    if not (workspace / "CMakeLists.txt").is_file():
        raise ValueError("allocator probe workspace has no CMakeLists.txt")
    if not (workspace / "p3_allocator_probe.cpp").is_file():
        raise ValueError("allocator probe workspace has no controller source")
    return {
        "func_list": [
            {
                "func_name": "p3_allocator_probe",
                "inputs_index": [0],
                "outputs_index": [0],
            }
        ],
        "input_num": 1,
        "output_num": 1,
        "target_bin": "libp3_allocator_probe.so",
        "workspace": str(workspace),
        "cmakelist_path": "CMakeLists.txt",
        "heavy_load": heavy_load,
        "compiler": str(toolchain_config),
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
                "flow_node_list": ["p3_allocator_probe_node"],
                "logic_device_list": "0:0:0:0",
            }
        ]
    }


def write(path: Path, value: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ascend-toolchain", type=Path, required=True)
    parser.add_argument("--function-output", type=Path, required=True)
    parser.add_argument("--toolchain-output", type=Path, required=True)
    parser.add_argument("--deploy-output", type=Path, required=True)
    parser.add_argument("--heavy-load", action="store_true")
    args = parser.parse_args()
    write(args.toolchain_output, toolchain_config(args.ascend_toolchain))
    write(
        args.function_output,
        function_config(args.workspace, args.toolchain_output, args.heavy_load),
    )
    write(args.deploy_output, deploy_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
