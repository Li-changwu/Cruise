#!/usr/bin/env python3
"""Inspect the narrow custom Device KV update AIR structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

from experiments.persistent_owner_graph_v2.kv_alias_probe.inspect_kv_alias import (
    _field,
    _source_name,
    node_blocks,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_graph_text(text: str) -> dict[str, object]:
    nodes = []
    for block in node_blocks(text):
        nodes.append(
            {
                "name": _field(block, "name"),
                "op": _field(block, "op"),
                "inputs": [
                    line.split('"', 2)[1]
                    for line in block.splitlines()
                    if line.lstrip().startswith('input: "')
                ],
            }
        )
    by_name = {str(node["name"]): node for node in nodes}
    counts = Counter(str(node["op"]) for node in nodes)
    updates = [node for node in nodes if node["op"] == "DeviceKvSlotUpdate"]
    graph_outputs = [node for node in nodes if node["op"] == "NetOutput"]
    input_ops: list[str | None] = []
    if len(updates) == 1 and len(updates[0]["inputs"]) == 2:
        input_ops = [
            str(by_name[name]["op"]) if name in by_name else None
            for name in (_source_name(value) for value in updates[0]["inputs"])
        ]
    output_sources = [
        _source_name(value)
        for node in graph_outputs
        for value in node["inputs"]
    ]
    report_only_output = (
        len(updates) == 1
        and len(graph_outputs) == 1
        and output_sources == [updates[0]["name"]]
    )
    passed = (
        len(updates) == 1
        and input_ops == ["Data", "Data"]
        and counts["RefData"] == 0
        and counts["TensorMove"] == 0
        and report_only_output
    )
    return {
        "gate": "V2-DEVICE-KV-UPDATE-STRUCTURE",
        "pass": passed,
        "node_count": len(nodes),
        "op_counts": dict(sorted(counts.items())),
        "device_kv_slot_update_count": len(updates),
        "update_input_ops": input_ops,
        "external_refdata_count": counts["RefData"],
        "tensor_move_count": counts["TensorMove"],
        "graph_output_sources": output_sources,
        "report_only_graph_output": report_only_output,
        "full_cache_graph_output": not report_only_output,
        "claim_boundary": (
            "One synthetic state-buffer update node only; no runtime lifetime, "
            "target KV, full Decoder, or P5 claim."
        ),
    }


def inspect_graph(path: Path) -> dict[str, object]:
    result = inspect_graph_text(path.read_text(encoding="utf-8"))
    result["graph_sha256"] = sha256(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = inspect_graph(args.graph.resolve(strict=True))
    except (OSError, ValueError) as exc:
        result = {
            "gate": "V2-DEVICE-KV-UPDATE-STRUCTURE",
            "pass": False,
            "error": str(exc),
        }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
