#!/usr/bin/env python3
"""Inspect the B4/K384 update-to-reader dependency graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node_blocks(text: str):
    block: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("node {"):
            if block:
                yield "".join(block)
            block = [line]
        elif block:
            block.append(line)
    if block:
        yield "".join(block)


def _field(block: str, name: str) -> str:
    match = re.search(rf'^\s+{re.escape(name)}: "([^"]+)"', block, re.MULTILINE)
    if not match:
        raise ValueError(f"node has no {name}")
    return match.group(1)


def _source_name(value: str) -> str:
    return value.lstrip("^").split(":", 1)[0]


def inspect_graph_text(text: str) -> dict[str, object]:
    nodes = []
    for block in node_blocks(text):
        nodes.append(
            {
                "name": _field(block, "name"),
                "op": _field(block, "op"),
                "inputs": re.findall(r'^\s+input: "([^"]+)"', block, re.MULTILINE),
            }
        )
    by_name = {str(node["name"]): node for node in nodes}
    counts = Counter(str(node["op"]) for node in nodes)
    updates = [node for node in nodes if node["op"] == "DevicePagedKvUpdate"]
    readers = [node for node in nodes if node["op"] == "DevicePagedKvRead"]
    outputs = [node for node in nodes if node["op"] == "NetOutput"]

    update_inputs: list[str] = []
    reader_inputs: list[str] = []
    if len(updates) == 1:
        update_inputs = [_source_name(value) for value in updates[0]["inputs"]]
    if len(readers) == 1:
        reader_inputs = [_source_name(value) for value in readers[0]["inputs"]]
    update_input_ops = [
        str(by_name[name]["op"]) if name in by_name else None for name in update_inputs
    ]
    reader_input_ops = [
        str(by_name[name]["op"]) if name in by_name else None for name in reader_inputs
    ]
    shared_state_input = (
        len(update_inputs) == 2
        and len(reader_inputs) == 3
        and update_inputs[0] == reader_inputs[0]
    )
    explicit_dependency = (
        len(updates) == 1
        and len(reader_inputs) == 3
        and reader_inputs[2] == updates[0]["name"]
    )
    output_inputs = (
        [_source_name(value) for value in outputs[0]["inputs"]]
        if len(outputs) == 1
        else []
    )
    report_only_output = (
        len(readers) == 1
        and len(output_inputs) == 1
        and output_inputs[0] == readers[0]["name"]
    )
    result = {
        "gate": "V2-KV-ORDER-STRUCTURE",
        "pass": bool(
            len(updates) == 1
            and len(readers) == 1
            and update_input_ops == ["Data", "Data"]
            and reader_input_ops == ["Data", "Data", "DevicePagedKvUpdate"]
            and shared_state_input
            and explicit_dependency
            and counts["RefData"] == 0
            and counts["TensorMove"] == 0
            and report_only_output
        ),
        "node_count": len(nodes),
        "op_counts": dict(sorted(counts.items())),
        "device_paged_kv_update_count": len(updates),
        "device_paged_kv_read_count": len(readers),
        "update_input_nodes": update_inputs,
        "update_input_ops": update_input_ops,
        "reader_input_nodes": reader_inputs,
        "reader_input_ops": reader_input_ops,
        "shared_state_input": shared_state_input,
        "explicit_update_to_reader_dependency": explicit_dependency,
        "external_refdata_count": counts["RefData"],
        "tensor_move_count": counts["TensorMove"],
        "report_only_graph_output": report_only_output,
        "full_state_graph_output": not report_only_output,
        "claim_boundary": (
            "AIR structure only; no target-hardware execution, Device-copy, "
            "attention, full-model, or performance claim."
        ),
    }
    return result


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
        result = {"gate": "V2-KV-ORDER-STRUCTURE", "pass": False, "error": str(exc)}
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
