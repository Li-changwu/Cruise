#!/usr/bin/env python3
"""Render compile and deployment configs for the mini FIA GraphPp probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--deploy-output", type=Path, required=True)
    parser.add_argument(
        "--kv-layout", choices=("dense", "pa-nz"), default="dense"
    )
    args = parser.parse_args()
    if args.kv_layout == "pa-nz":
        key_shape = [12, 4, 8, 128, 16]
        auxiliary = {"data_type": "DT_INT32", "shape": [4, 3]}
    else:
        key_shape = [4, 4, 384, 128]
        auxiliary = {"data_type": "DT_BOOL", "shape": [4, 1, 1, 384]}
    write(
        args.graph_output,
        {
            "inputs_tensor_desc": [
                {"data_type": "DT_BFLOAT16", "shape": [4, 28, 1, 128]},
                {"data_type": "DT_BFLOAT16", "shape": key_shape},
                {"data_type": "DT_BFLOAT16", "shape": key_shape},
                auxiliary,
            ]
        },
    )
    write(
        args.deploy_output,
        {
            "batch_deploy_info": [
                {
                    "flow_node_list": ["mini_fia_node"],
                    "logic_device_list": "0:0:0:0",
                }
            ]
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
