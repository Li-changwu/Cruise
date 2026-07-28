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


def field(block: str, name: str) -> str:
    match = re.search(rf'^\s+{name}: "([^"]+)"', block, re.MULTILINE)
    if not match:
        raise ValueError(f"node has no {name}")
    return match.group(1)


def descriptor_shape(block: str, descriptor: str) -> tuple[int, ...]:
    match = re.search(
        rf'key: "{re.escape(descriptor)}".*?s: \'(.*?)\'\n', block, re.DOTALL
    )
    if not match:
        return ()
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


def bool_attr(block: str, name: str) -> bool:
    return bool(re.search(rf'key: "{name}".*?b: true\\n', block, re.DOTALL))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nodes = [(field(block, "name"), field(block, "op"), block) for block in node_blocks(args.graph)]
    node_ops = {name: op for name, op, _ in nodes}
    details = []
    for name, op, block in nodes:
        if op != "MatMulV2" or name.startswith("LinearTransposeX2"):
            continue
        inputs = re.findall(r'^\s+input: "([^"]*)"', block, re.MULTILINE)
        x2_source = inputs[1] if len(inputs) > 1 else None
        x2_node = x2_source.split(":", 1)[0] if x2_source else None
        details.append(
            {
                "name": name,
                "x1_shape": list(descriptor_shape(block, "[i]x1")),
                "x2_shape": list(descriptor_shape(block, "[i]x2")),
                "x2_source": x2_source,
                "x2_source_op": node_ops.get(x2_node),
                "transpose_x1": bool_attr(block, "transpose_x1"),
                "transpose_x2": bool_attr(block, "transpose_x2"),
                "input_count": len(inputs),
            }
        )

    signatures = Counter(
        (
            tuple(item["x1_shape"]),
            tuple(item["x2_shape"]),
            item["x2_source_op"],
            item["transpose_x1"],
            item["transpose_x2"],
            item["input_count"],
        )
        for item in details
    )
    result = {
        "remaining_matmulv2_count": len(details),
        "signatures": [
            {
                "count": count,
                "x1_shape": list(signature[0]),
                "x2_shape": list(signature[1]),
                "x2_source_op": signature[2],
                "transpose_x1": signature[3],
                "transpose_x2": signature[4],
                "input_count": signature[5],
            }
            for signature, count in sorted(signatures.items(), key=lambda item: str(item[0]))
        ],
        "nodes": details,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("remaining_matmulv2_count", "signatures")}, indent=2))


if __name__ == "__main__":
    main()
