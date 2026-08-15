#!/usr/bin/env python3
"""Inspect whether ScatterPaKvCache updates direct RefData KV inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    op_counts = Counter(str(node["op"]) for node in nodes)
    scatter_nodes = [node for node in nodes if node["op"] == "ScatterPaKvCache"]
    cache_inputs: list[str] = []
    cache_input_ops: list[str | None] = []
    if len(scatter_nodes) == 1:
        inputs = scatter_nodes[0]["inputs"]
        if isinstance(inputs, list) and len(inputs) >= 5:
            cache_inputs = [_source_name(inputs[1]), _source_name(inputs[4])]
            cache_input_ops = [
                str(by_name[name]["op"]) if name in by_name else None
                for name in cache_inputs
            ]

    kv_refdata_alias = cache_input_ops == ["RefData", "RefData"]
    no_kv_tensor_move = bool(cache_input_ops) and "TensorMove" not in cache_input_ops
    result = {
        "gate": "V2-KV-ALIAS-STRUCTURE",
        "pass": bool(
            len(scatter_nodes) == 1
            and kv_refdata_alias
            and no_kv_tensor_move
        ),
        "node_count": len(nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "scatter_pa_kv_cache_count": len(scatter_nodes),
        "kv_cache_input_nodes": cache_inputs,
        "kv_cache_input_ops": cache_input_ops,
        "kv_refdata_alias": kv_refdata_alias,
        "no_kv_tensor_move": no_kv_tensor_move,
        "claim_boundary": (
            "AIR graph structure only; no ordinary Graph, GraphPp, Device-copy, "
            "shared-lease, full-model, or performance claim."
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
        result = {
            "gate": "V2-KV-ALIAS-STRUCTURE",
            "pass": False,
            "error": str(exc),
        }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
