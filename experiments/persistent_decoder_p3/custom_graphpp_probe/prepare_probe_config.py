#!/usr/bin/env python3
"""Render GraphPp compile and deployment configs for the custom-op probe."""

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
    args = parser.parse_args()
    write(
        args.graph_output,
        {
            "build_options": {
                "ge.modelFileNamePrefix": "cruise_bf16_materialize_probe"
            },
            "inputs_tensor_desc": [
                {"data_type": "DT_BFLOAT16", "shape": [1, 1, 18944]}
            ],
        },
    )
    write(
        args.deploy_output,
        {
            "batch_deploy_info": [
                {
                    "flow_node_list": ["bf16_materialize_node"],
                    "logic_device_list": "0:0:0:0",
                }
            ]
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
