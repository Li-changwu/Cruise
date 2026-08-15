#!/usr/bin/env python3
"""Render the DataFlow compile configuration for the P0 controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_config(workspace: Path, toolchain_config: Path) -> dict[str, object]:
    workspace = workspace.resolve(strict=True)
    toolchain_config = toolchain_config.resolve(strict=True)
    if not (workspace / "CMakeLists.txt").is_file():
        raise ValueError("controller workspace has no CMakeLists.txt")
    if not (workspace / "persistent_control_p0.cpp").is_file():
        raise ValueError("controller workspace has no P0 source")
    return {
        "func_list": [
            {
                "func_name": "persistent_control_p0",
                "inputs_index": [0],
                "outputs_index": [0],
                "stream_input": True,
            }
        ],
        "input_num": 1,
        "output_num": 1,
        "target_bin": "libpersistent_control_p0.so",
        "workspace": str(workspace),
        "cmakelist_path": "CMakeLists.txt",
        "heavy_load": False,
        "compiler": str(toolchain_config),
    }


def make_toolchain_config(ascend_toolchain: Path) -> dict[str, object]:
    ascend_toolchain = ascend_toolchain.resolve(strict=True)
    if not ascend_toolchain.is_file():
        raise ValueError("Ascend FunctionPp toolchain is not a file")
    return {
        "compiler": [
            {
                "resource_type": "Ascend",
                "toolchain": str(ascend_toolchain),
            }
        ]
    }


def make_deploy_config() -> dict[str, object]:
    return {
        "batch_deploy_info": [
            {
                "flow_node_list": ["persistent_control_p0_node"],
                "logic_device_list": "0:0:0:0",
            }
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ascend-toolchain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toolchain-output", type=Path, required=True)
    parser.add_argument("--deploy-output", type=Path, required=True)
    args = parser.parse_args()
    toolchain_output = args.toolchain_output.resolve()
    toolchain_output.parent.mkdir(parents=True, exist_ok=True)
    toolchain_output.write_text(
        json.dumps(make_toolchain_config(args.ascend_toolchain), indent=2) + "\n",
        encoding="utf-8",
    )
    payload = make_config(args.workspace, toolchain_output)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    deploy_output = args.deploy_output.resolve()
    deploy_output.parent.mkdir(parents=True, exist_ok=True)
    deploy_output.write_text(
        json.dumps(make_deploy_config(), indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
