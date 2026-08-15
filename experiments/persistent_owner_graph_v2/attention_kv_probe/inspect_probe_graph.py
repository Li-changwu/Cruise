#!/usr/bin/env python3
"""Inspect the explicit update-to-FIA dependency and no-full-KV-output graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def _blocks(text: str):
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
    match = re.search(rf'^\s+{name}: "([^"]+)"', block, re.MULTILINE)
    if not match:
        raise ValueError(f"node has no {name}")
    return match.group(1)


def _source(value: str) -> str:
    return value.lstrip("^").split(":", 1)[0]


def inspect_graph_text(text: str) -> dict[str, object]:
    nodes = [
        {
            "name": _field(block, "name"),
            "op": _field(block, "op"),
            "inputs": re.findall(r'^\s+input: "([^"]*)"', block, re.MULTILINE),
        }
        for block in _blocks(text)
    ]
    by_name = {str(node["name"]): node for node in nodes}
    counts = Counter(str(node["op"]) for node in nodes)
    updates = [node for node in nodes if node["op"] == "DevicePagedKvUpdate"]
    orders = [node for node in nodes if node["op"] == "DeviceQueryAfterKvUpdate"]
    fias = [node for node in nodes if node["op"] == "FusedInferAttentionScore"]
    outputs = [node for node in nodes if node["op"] == "NetOutput"]

    update_inputs = [_source(item) for item in updates[0]["inputs"]] if len(updates) == 1 else []
    order_inputs = [_source(item) for item in orders[0]["inputs"]] if len(orders) == 1 else []
    fia_inputs = [_source(item) if item else "" for item in fias[0]["inputs"]] if len(fias) == 1 else []
    output_inputs = [_source(item) for item in outputs[0]["inputs"]] if len(outputs) == 1 else []
    input_ops = {
        name: str(by_name[name]["op"]) if name in by_name else None
        for name in set(update_inputs + order_inputs + [item for item in fia_inputs if item])
    }
    shared_key_value = (
        len(update_inputs) == 3
        and len(fia_inputs) == 31
        and fia_inputs[1:3] == update_inputs[0:2]
    )
    explicit_dependency = (
        len(order_inputs) == 2
        and len(updates) == 1
        and order_inputs[1] == updates[0]["name"]
        and len(fia_inputs) == 31
        and len(orders) == 1
        and fia_inputs[0] == orders[0]["name"]
    )
    compact_outputs = (
        len(output_inputs) == 2
        and len(fias) == 1
        and len(updates) == 1
        and output_inputs == [fias[0]["name"], updates[0]["name"]]
    )
    data_inputs_only = all(input_ops.get(name) == "Data" for name in update_inputs)
    result = {
        "gate": "V2-KV-ATTENTION-STRUCTURE",
        "pass": bool(
            len(updates) == 1
            and len(orders) == 1
            and len(fias) == 1
            and data_inputs_only
            and shared_key_value
            and explicit_dependency
            and compact_outputs
            and counts["RefData"] == 0
            and counts["TensorMove"] == 0
        ),
        "node_count": len(nodes),
        "op_counts": dict(sorted(counts.items())),
        "update_inputs": update_inputs,
        "order_inputs": order_inputs,
        "fia_inputs": fia_inputs,
        "shared_update_and_fia_kv_inputs": shared_key_value,
        "explicit_update_to_fia_dependency": explicit_dependency,
        "external_refdata_count": counts["RefData"],
        "tensor_move_count": counts["TensorMove"],
        "compact_attention_and_ticket_outputs": compact_outputs,
        "full_kv_graph_output": not compact_outputs,
        "claim_boundary": "AIR structure only; no execution or copy claim.",
    }
    return result


def inspect_graph(path: Path) -> dict[str, object]:
    result = inspect_graph_text(path.read_text(encoding="utf-8"))
    result["graph_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = inspect_graph(args.graph.resolve(strict=True))
    except (OSError, ValueError) as error:
        result = {"gate": "V2-KV-ATTENTION-STRUCTURE", "pass": False, "error": str(error)}
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
