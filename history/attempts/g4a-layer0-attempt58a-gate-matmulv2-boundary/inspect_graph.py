#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def node_blocks(path: Path):
    block = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("node {"):
                if block:
                    yield "".join(block)
                block = [line]
            elif block:
                block.append(line)
    if block:
        yield "".join(block)


def node_name(block: str) -> str:
    match = re.search(r'^\s+name: "([^"]+)"', block, re.MULTILINE)
    if not match:
        raise ValueError("node has no name")
    return match.group(1)


def node_op(block: str) -> str:
    match = re.search(r'^\s+op: "([^"]+)"', block, re.MULTILINE)
    if not match:
        raise ValueError("node has no op")
    return match.group(1)


def descriptor_shape(block: str, descriptor: str) -> list[int]:
    match = re.search(
        rf'key: "{re.escape(descriptor)}".*?s: \'(.*?)\'\n', block, re.DOTALL
    )
    if not match:
        raise ValueError(f"missing descriptor {descriptor}")
    value = match.group(1)
    dimensions = [int(item) for item in re.findall(r"dim: (-?\d+)", value)]
    if dimensions:
        return dimensions
    meta = re.search(r"shape=torch.Size\(\[([0-9, ]*)\]\)", value)
    if not meta:
        return []
    return [int(item.strip()) for item in meta.group(1).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nodes = [(node_name(block), node_op(block), block) for block in node_blocks(args.graph)]
    counts = Counter(op for _, op, _ in nodes)
    gate_nodes = [item for item in nodes if item[0].startswith("GateMatMulV2")]
    gate = gate_nodes[0] if len(gate_nodes) == 1 else None
    gate_valid = False
    gate_detail = None
    if gate is not None:
        name, op, block = gate
        gate_detail = {
            "name": name,
            "op": op,
            "x1_shape": descriptor_shape(block, "[i]x1"),
            "x2_shape": descriptor_shape(block, "[i]x2"),
            "output_shape": descriptor_shape(block, "[o]y"),
        }
        gate_valid = gate_detail == {
            "name": "GateMatMulV2",
            "op": "MatMulV2",
            "x1_shape": [1, 3584],
            "x2_shape": [3584, 18944],
            "output_shape": [1, 1, 18944],
        }

    required_counts = {
        "ExactQk": 1,
        "Bf16Barrier": 1,
        "Bf16Materialize": 1,
        "MatMulV2": 4,
        "MatMul": 3,
    }
    counts_valid = all(counts[name] == expected for name, expected in required_counts.items())
    result = {
        "valid": gate_valid and counts_valid,
        "gate_node_valid": gate_valid,
        "gate_node": gate_detail,
        "required_counts": required_counts,
        "observed_counts": {name: counts[name] for name in required_counts},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["valid"]:
        raise SystemExit(91)


if __name__ == "__main__":
    main()
