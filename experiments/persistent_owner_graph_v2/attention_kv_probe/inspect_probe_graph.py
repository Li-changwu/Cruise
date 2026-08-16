#!/usr/bin/env python3
"""Inspect the explicit update-to-FIA dependency and no-full-KV-output graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_DATA_INPUTS = (
    ("key_cache", "DT_BF16", [12, 4, 8, 128, 16]),
    ("value_cache", "DT_BF16", [12, 4, 8, 128, 16]),
    ("metadata", "DT_INT32", [5]),
    ("query", "DT_BF16", [4, 28, 1, 128]),
    ("mask", "DT_BOOL", [4, 1, 1, 384]),
    ("block_table", "DT_INT32", [4, 3]),
)


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


def _data_contract(block: str) -> tuple[int, str, list[int]]:
    index = re.search(
        r'key: "index"\s+value \{\s+s: \'i: (\d+)\\n\'', block
    )
    dtypes = re.findall(r"dtype: (DT_[A-Z0-9]+)", block)
    dimensions = [int(value) for value in re.findall(r"dim: (-?\d+)", block)]
    if index is None or not dtypes or any(dtype != dtypes[0] for dtype in dtypes):
        raise ValueError("Data node has no unique index or dtype")
    half = len(dimensions) // 2
    if len(dimensions) % 2 == 0 and dimensions[:half] == dimensions[half:]:
        dimensions = dimensions[:half]
    return int(index.group(1)), dtypes[0], dimensions


def inspect_graph_text(text: str) -> dict[str, object]:
    nodes = [
        {
            "name": _field(block, "name"),
            "op": _field(block, "op"),
            "inputs": re.findall(r'^\s+input: "([^"]*)"', block, re.MULTILINE),
            "block": block,
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
    role_sources = {}
    if len(update_inputs) == 3:
        role_sources.update(
            {
                "key_cache": update_inputs[0],
                "value_cache": update_inputs[1],
                "metadata": update_inputs[2],
            }
        )
    if len(order_inputs) == 2:
        role_sources["query"] = order_inputs[0]
    if len(fia_inputs) == 31:
        role_sources["mask"] = fia_inputs[4]
        role_sources["block_table"] = fia_inputs[14]
    data_input_abi = []
    for role, source in role_sources.items():
        node = by_name.get(source)
        if node is None or node["op"] != "Data":
            continue
        index, dtype, shape = _data_contract(str(node["block"]))
        data_input_abi.append(
            {
                "index": index,
                "role": role,
                "node": source,
                "data_type": dtype,
                "shape": shape,
            }
        )
    data_input_abi.sort(key=lambda item: int(item["index"]))
    expected_abi = [
        {
            "index": index,
            "role": role,
            "data_type": dtype,
            "shape": shape,
        }
        for index, (role, dtype, shape) in enumerate(EXPECTED_DATA_INPUTS)
    ]
    comparable_abi = [
        {
            key: item[key]
            for key in ("index", "role", "data_type", "shape")
        }
        for item in data_input_abi
    ]
    data_input_abi_pass = comparable_abi == expected_abi
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
            and data_input_abi_pass
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
        "data_input_abi": data_input_abi,
        "data_input_abi_pass": data_input_abi_pass,
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
