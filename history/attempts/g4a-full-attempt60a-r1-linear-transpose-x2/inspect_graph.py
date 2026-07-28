#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_OP_COUNTS = {
    "MatMul": 0,
    "MatMulV2": 197,
    "ExactQk": 28,
    "Bf16Barrier": 28,
    "BatchMatMul": 29,
}
EXPECTED_WEIGHT_SHAPES = {
    (3584, 3584): 28,
    (18944, 3584): 56,
    (3584, 18944): 28,
    (152064, 3584): 1,
}


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


def descriptor_shape(block: str, descriptor: str) -> tuple[int, ...]:
    match = re.search(
        rf'key: "{re.escape(descriptor)}".*?s: \'(.*?)\'\n', block, re.DOTALL
    )
    if not match:
        raise ValueError(f"missing descriptor {descriptor}")
    value = match.group(1)
    dimensions = tuple(int(item) for item in re.findall(r"dim: (-?\d+)", value))
    if dimensions:
        return dimensions
    meta = re.search(r"shape=torch.Size\(\[([0-9, ]*)\]\)", value)
    if not meta:
        return ()
    return tuple(
        int(item.strip()) for item in meta.group(1).split(",") if item.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nodes = [(node_name(block), node_op(block), block) for block in node_blocks(args.graph)]
    node_ops = {name: op for name, op, _ in nodes}
    op_counts = Counter(op for _, op, _ in nodes)
    linear_nodes = [item for item in nodes if item[0].startswith("LinearTransposeX2")]
    weight_shapes = Counter()
    details = []
    for name, op, block in linear_nodes:
        inputs = re.findall(r'^\s+input: "([^"]*)"', block, re.MULTILINE)
        x2_source = inputs[1] if len(inputs) > 1 else None
        x2_node = x2_source.split(":", 1)[0] if x2_source else None
        x1_shape = descriptor_shape(block, "[i]x1")
        x2_shape = descriptor_shape(block, "[i]x2")
        weight_shapes[x2_shape] += 1
        transpose_x2 = bool(
            re.search(r'key: "transpose_x2".*?b: true\\n', block, re.DOTALL)
        )
        source_op = node_ops.get(x2_node)
        valid = (
            op == "MatMulV2"
            and transpose_x2
            and x2_source is not None
            and x2_source.startswith("arg")
            and source_op != "Transpose"
            and x2_shape in EXPECTED_WEIGHT_SHAPES
            and x1_shape == (1, x2_shape[1])
        )
        details.append(
            {
                "name": name,
                "op": op,
                "x1_shape": list(x1_shape),
                "x2_shape": list(x2_shape),
                "x2_source": x2_source,
                "x2_source_op": source_op,
                "transpose_x2": transpose_x2,
                "valid": valid,
            }
        )

    observed_counts = {name: op_counts[name] for name in EXPECTED_OP_COUNTS}
    observed_weight_shapes = {
        "x".join(str(value) for value in shape): count
        for shape, count in sorted(weight_shapes.items())
    }
    expected_weight_shapes = {
        "x".join(str(value) for value in shape): count
        for shape, count in EXPECTED_WEIGHT_SHAPES.items()
    }
    result = {
        "valid": (
            len(linear_nodes) == 113
            and all(item["valid"] for item in details)
            and weight_shapes == Counter(EXPECTED_WEIGHT_SHAPES)
            and observed_counts == EXPECTED_OP_COUNTS
        ),
        "linear_node_count": len(linear_nodes),
        "linear_nodes_all_valid": all(item["valid"] for item in details),
        "expected_weight_shapes": expected_weight_shapes,
        "observed_weight_shapes": observed_weight_shapes,
        "expected_op_counts": EXPECTED_OP_COUNTS,
        "observed_op_counts": observed_counts,
        "linear_nodes": details,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["valid"]:
        raise SystemExit(91)


if __name__ == "__main__":
    main()
