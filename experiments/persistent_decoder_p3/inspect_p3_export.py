#!/usr/bin/env python3
"""Audit the actual P3 AIR graph ABI and required decoder structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


CACHE_SIGNATURE = ("DT_BF16", (28, 12, 128, 4, 128))
CONTROL_VECTOR_SIGNATURE = ("DT_INT32", (4,))
UNIQUE_INPUTS = {
    ("DT_INT64", (4, 1)): "token_id",
    ("DT_INT64", (4,)): "position",
    ("DT_INT32", (4, 1)): "sequence_length",
    ("DT_INT32", (4, 3)): "block_table",
}
EXPECTED_INPUTS = {
    "token_id": ("DT_INT64", (4, 1)),
    "position": ("DT_INT64", (4,)),
    "sequence_length": ("DT_INT32", (4, 1)),
    "block_table": ("DT_INT32", (4, 3)),
    "slot_mapping": CONTROL_VECTOR_SIGNATURE,
    "active_mask": CONTROL_VECTOR_SIGNATURE,
    "key_cache": CACHE_SIGNATURE,
    "value_cache": CACHE_SIGNATURE,
}
EXPECTED_OUTPUTS = (
    ("DT_INT64", (4, 1), "token_id"),
    ("DT_BF16", (28, 12, 128, 4, 128), "key_cache"),
    ("DT_BF16", (28, 12, 128, 4, 128), "value_cache"),
    ("DT_INT64", (4,), "next_position"),
)
REQUIRED_OP_COUNTS = {
    "Data": 8,
    "NetOutput": 1,
    "FileConstant": 202,
    "ArgMaxV2": 1,
    "FusedInferAttentionScore": 28,
    "RmsNorm": 1,
    "AddRmsNorm": 56,
    "RotaryMul": 56,
    "MatMul": 85,
    "MatMulV2": 28,
    "Swish": 28,
}
FORBIDDEN_OPS = ("SoftmaxV2", "BatchMatMul")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def descriptor(text: str) -> tuple[str, tuple[int, ...]]:
    match = re.search(
        r"dtype: ([A-Z0-9_]+)\\nshape \{(.*?)\}\\nlayout:", text, re.DOTALL
    )
    if not match:
        raise ValueError("tensor descriptor parse failed")
    dimensions = tuple(
        int(value) for value in re.findall(r"dim: (-?\d+)", match.group(2))
    )
    return match.group(1), dimensions


def descriptor_for(block: str, key: str) -> tuple[str, tuple[int, ...]]:
    match = re.search(
        rf'key: "{re.escape(key)}".*?s: \'(name: ".*?layout:.*?\\n)\'',
        block,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"missing tensor descriptor: {key}")
    return descriptor(match.group(1))


def parse_inputs(nodes: list[tuple[str, str, str]]) -> list[dict]:
    graph_text = "".join(block for _, _, block in nodes)
    inputs = []
    for name, op, block in nodes:
        if op != "Data":
            continue
        index = re.search(r'key: "index".*?s: \'i: (\d+)\\n\'', block, re.DOTALL)
        if not index:
            raise ValueError(f"Data node has no index: {name}")
        dtype, shape = descriptor_for(block, "[i]x")
        inputs.append(
            {
                "index": int(index.group(1)),
                "node_name": name,
                "dtype": dtype,
                "shape": list(shape),
                "direct_consumer_count": graph_text.count(f'input: "{name}:0"'),
            }
        )
    inputs.sort(key=lambda item: item["index"])

    cache_index = 0
    for item in inputs:
        signature = (item["dtype"], tuple(item["shape"]))
        if signature == CACHE_SIGNATURE:
            item["semantic"] = "key_cache" if cache_index == 0 else "value_cache"
            cache_index += 1
        elif signature == CONTROL_VECTOR_SIGNATURE:
            consumers = item["direct_consumer_count"]
            if consumers == 224:
                item["semantic"] = "slot_mapping"
            elif consumers == 225:
                item["semantic"] = "active_mask"
            else:
                item["semantic"] = "unknown"
        else:
            item["semantic"] = UNIQUE_INPUTS.get(signature, "unknown")
    return inputs


def parse_outputs(nodes: list[tuple[str, str, str]]) -> list[dict]:
    blocks = [block for _, op, block in nodes if op == "NetOutput"]
    if len(blocks) != 1:
        raise ValueError(f"expected one NetOutput, observed {len(blocks)}")
    outputs = []
    for match in re.finditer(
        r'key: "\[i\]input(\d+)".*?s: \'(name: "input\d+".*?layout:.*?\\n)\'',
        blocks[0],
        re.DOTALL,
    ):
        dtype, shape = descriptor(match.group(2))
        outputs.append(
            {"index": int(match.group(1)), "dtype": dtype, "shape": list(shape)}
        )
    outputs.sort(key=lambda item: item["index"])
    for item, expected in zip(outputs, EXPECTED_OUTPUTS):
        item["semantic"] = expected[2]
    return outputs


def parse_file_constants(
    nodes: list[tuple[str, str, str]], export_dir: Path
) -> list[dict]:
    result = []
    for name, op, block in nodes:
        if op != "FileConstant":
            continue
        match = re.search(
            r'key: "file_path".*?s: \'s: "([^"]+)"\\n\'',
            block,
            re.DOTALL,
        )
        if not match:
            raise ValueError(f"FileConstant has no file_path: {name}")
        path = Path(match.group(1))
        direct_child = path.parent == export_dir
        result.append(
            {
                "node_name": name,
                "name": path.name,
                "path": str(path),
                "direct_child": direct_child,
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    return result


def select_air(export_dir: Path) -> Path:
    candidates = (
        export_dir / "qwen_b4_p3_decoder_step.runtime.air",
        export_dir / "qwen_b4_p3_decoder_step.air",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(
            f"expected exactly one P3 AIR in {export_dir}, observed {len(existing)}"
        )
    return existing[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    export_dir = args.export_dir.resolve(strict=True)
    graph = export_dir / "dynamo.pbtxt"
    air = select_air(export_dir)
    nodes = [(node_name(block), node_op(block), block) for block in node_blocks(graph)]
    inputs = parse_inputs(nodes)
    outputs = parse_outputs(nodes)
    file_constants = parse_file_constants(nodes, export_dir)
    op_counts = Counter(op for _, op, _ in nodes)
    observed_required_ops = {
        name: op_counts[name] for name in REQUIRED_OP_COUNTS
    }

    input_valid = (
        len(inputs) == len(EXPECTED_INPUTS)
        and [item["index"] for item in inputs] == list(range(len(inputs)))
        and {item["semantic"] for item in inputs} == set(EXPECTED_INPUTS)
        and all(
            (item["dtype"], tuple(item["shape"]))
            == EXPECTED_INPUTS[item["semantic"]]
            for item in inputs
        )
    )
    output_valid = len(outputs) == len(EXPECTED_OUTPUTS) and all(
        item["index"] == index
        and (item["dtype"], tuple(item["shape"])) == expected[:2]
        for index, (item, expected) in enumerate(zip(outputs, EXPECTED_OUTPUTS))
    )
    file_constants_valid = (
        len(file_constants) == REQUIRED_OP_COUNTS["FileConstant"]
        and len({item["name"] for item in file_constants}) == len(file_constants)
        and all(item["direct_child"] and item["exists"] for item in file_constants)
    )
    forbidden_op_counts = {name: op_counts[name] for name in FORBIDDEN_OPS}
    structure_valid = (
        observed_required_ops == REQUIRED_OP_COUNTS
        and not any(forbidden_op_counts.values())
        and file_constants_valid
    )
    result = {
        "gate": "P3-AIR384-actual-graph-audit",
        "pass": bool(air.is_file() and input_valid and output_valid and structure_valid),
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph),
        "input_valid": input_valid,
        "output_valid": output_valid,
        "structure_valid": structure_valid,
        "file_constants_valid": file_constants_valid,
        "external_file_count": len(file_constants),
        "external_file_bytes": sum(int(item["bytes"]) for item in file_constants),
        "external_files": [item["name"] for item in file_constants],
        "native_feed_order": [item["semantic"] for item in inputs],
        "inputs": inputs,
        "outputs": outputs,
        "required_op_counts": observed_required_ops,
        "forbidden_op_counts": forbidden_op_counts,
        "semantic_rule": (
            "Unique dtype/shape signatures identify four control inputs; "
            "cache Data order identifies key then value; INT32-vector consumer "
            "counts distinguish slot_mapping from active_mask. Native recurrence "
            "must separately validate cache semantics."
        ),
        "claim_boundary": (
            "Actual AIR graph ABI and decoder structure only; no native GE "
            "recurrence, Persistent Owner, or service-performance claim."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
