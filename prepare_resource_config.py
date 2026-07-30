#!/usr/bin/env python3
"""Prepare a single-device DataFlow resource configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def prepare_resource_config(
    template: Path, physical_npu: int, deploy_root: Path
) -> dict[str, Any]:
    if physical_npu < 0:
        raise ValueError("physical NPU ID must be non-negative")
    payload = json.loads(template.read_text(encoding="utf-8"))
    clusters = payload.get("cluster")
    if not isinstance(clusters, list) or len(clusters) != 1:
        raise ValueError("resource template must contain exactly one cluster")
    nodes = clusters[0].get("cluster_nodes")
    if not isinstance(nodes, list) or len(nodes) != 1:
        raise ValueError("resource template must contain exactly one cluster node")
    node = nodes[0]
    items = node.get("item_list")
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError("resource template must contain exactly one device item")
    if not isinstance(items[0], dict) or items[0].get("item_id") != 0:
        raise ValueError("resource template device item must have logical item_id 0")
    items[0]["device_id"] = physical_npu
    node["deploy_res_path"] = str(deploy_root.resolve())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--physical-npu", type=int, required=True)
    parser.add_argument("--deploy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = prepare_resource_config(
        args.template.resolve(strict=True), args.physical_npu, args.deploy_root
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
