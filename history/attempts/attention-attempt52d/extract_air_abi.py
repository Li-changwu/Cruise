#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


EXPECTED_INPUTS = (
    {"index": 0, "semantic": "key_cache", "dtype": "DT_FLOAT16", "shape": [1, 4, 8, 128]},
    {"index": 1, "semantic": "hidden_table", "dtype": "DT_FLOAT16", "shape": [4, 3584]},
    {"index": 2, "semantic": "position", "dtype": "DT_INT64", "shape": [1]},
    {"index": 3, "semantic": "value_cache", "dtype": "DT_FLOAT16", "shape": [1, 4, 8, 128]},
    {"index": 4, "semantic": "explicit_tiling", "dtype": "DT_UINT8", "shape": [72]},
)
EXPECTED_OUTPUT_DTYPES = (
    "DT_FLOAT16", "DT_FLOAT16", "DT_FLOAT16", "DT_INT64",
    "DT_FLOAT16", "DT_FLOAT16", "DT_FLOAT", "DT_FLOAT16",
    "DT_FLOAT16", "DT_FLOAT16", "DT_FLOAT16", "DT_BF16",
    "DT_BF16", "DT_BF16", "DT_BF16", "DT_BF16", "DT_BF16",
)
EXPECTED_OUTPUT_SHAPES = (
    [1, 1, 3584], [1, 4, 8, 128], [1, 4, 8, 128], [1],
    [1, 1, 512], [1, 4, 1, 128], [1, 28, 1, 8], [1, 1, 3584],
    [1, 28, 1, 128], [1, 1, 1, 128], [1, 1, 1, 128], [1, 1, 3584],
    [1, 1, 512], [1, 28, 1, 128], [1, 4, 1, 128], [1, 4, 8, 128],
    [1, 28, 1, 8],
)


def target_blocks(path: Path):
    prefix, block, in_node, decision = [], [], False, None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("node {"):
                if in_node and decision:
                    yield "".join(block)
                prefix, block, in_node, decision = [line], [], True, None
                continue
            if not in_node:
                continue
            if decision is True:
                block.append(line)
            elif decision is None:
                prefix.append(line)
                match = re.match(r'^\s+op: "([^"]+)"', line)
                if match:
                    decision = match.group(1) in {"Data", "NetOutput"}
                    block = prefix if decision else []
        if in_node and decision:
            yield "".join(block)


def descriptor(text: str) -> tuple[str, list[int]]:
    match = re.search(r"dtype: ([A-Z0-9_]+)\\nshape \{(.*?)\}\\nlayout:", text, re.DOTALL)
    if not match:
        raise ValueError("tensor descriptor parse failed")
    return match.group(1), [int(value) for value in re.findall(r"dim: (-?\d+)", match.group(2))]


def parse_inputs(path: Path) -> list[dict]:
    result = []
    for block in target_blocks(path):
        if 'op: "Data"' not in block:
            continue
        name = re.search(r'^\s+name: "([^"]+)"', block, re.MULTILINE)
        index = re.search(r'key: "index".*?s: \'i: (\d+)\\n\'', block, re.DOTALL)
        desc = re.search(r'key: "\[i\]x".*?s: \'(name: "x".*?layout:.*?\\n)\'', block, re.DOTALL)
        if not name or not index or not desc:
            raise ValueError("Data node parse failed")
        arg = re.search(r"arg(\d+)_", name.group(1))
        if not arg:
            raise ValueError(f"unknown Data node name {name.group(1)}")
        dtype, shape = descriptor(desc.group(1))
        ordinal = int(arg.group(1))
        result.append({
            "index": int(index.group(1)),
            "node_name": name.group(1),
            "arg_ordinal": ordinal,
            "dtype": dtype,
            "shape": shape,
        })
    return sorted(result, key=lambda item: item["index"])


def parse_outputs(path: Path) -> list[dict]:
    block = next(value for value in target_blocks(path) if 'op: "NetOutput"' in value)
    result = []
    for match in re.finditer(
        r'key: "\[i\]input(\d+)".*?s: \'(name: "input\d+".*?layout:.*?\\n)\'',
        block,
        re.DOTALL,
    ):
        dtype, shape = descriptor(match.group(2))
        result.append({"index": int(match.group(1)), "dtype": dtype, "shape": shape})
    return sorted(result, key=lambda item: item["index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = parse_inputs(args.graph)
    outputs = parse_outputs(args.graph)
    for item, expected in zip(inputs, EXPECTED_INPUTS):
        item["semantic"] = expected["semantic"]
    input_valid = len(inputs) == len(EXPECTED_INPUTS) and all(
        item["index"] == expected["index"]
        and item["dtype"] == expected["dtype"]
        and item["shape"] == expected["shape"]
        for item, expected in zip(inputs, EXPECTED_INPUTS)
    )
    output_valid = len(outputs) == len(EXPECTED_OUTPUT_DTYPES) and all(
        item["index"] == index
        and item["dtype"] == EXPECTED_OUTPUT_DTYPES[index]
        and item["shape"] == EXPECTED_OUTPUT_SHAPES[index]
        for index, item in enumerate(outputs)
    )
    result = {
        "valid": input_valid and output_valid,
        "input_valid": input_valid,
        "output_valid": output_valid,
        "inputs": inputs,
        "outputs": outputs,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    if not result["valid"]:
        raise SystemExit(9)


if __name__ == "__main__":
    main()
