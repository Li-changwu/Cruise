#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


EXPECTED_INPUTS = {
    ("DT_BF16", (1, 1, 3584)): "hidden",
    ("DT_INT64", (1,)): "position",
    ("DT_INT32", (1, 1)): "sequence_length",
    ("DT_INT32", (1, 2)): "block_table",
    ("DT_INT32", (1,)): "slot_mapping",
    ("DT_UINT8", (72,)): "explicit_tiling",
}
CACHE_SIGNATURE = ("DT_BF16", (2, 128, 4, 128))


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


def descriptor(text: str) -> tuple[str, tuple[int, ...]]:
    match = re.search(r"dtype: ([A-Z0-9_]+)\\nshape \{(.*?)\}\\nlayout:", text, re.DOTALL)
    if not match:
        raise ValueError("tensor descriptor parse failed")
    return match.group(1), tuple(int(value) for value in re.findall(r"dim: (-?\d+)", match.group(2)))


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
        dtype, shape = descriptor(desc.group(1))
        result.append(
            {"index": int(index.group(1)), "node_name": name.group(1), "dtype": dtype, "shape": list(shape)}
        )
    result.sort(key=lambda item: item["index"])
    cache_seen = 0
    for item in result:
        signature = (item["dtype"], tuple(item["shape"]))
        if signature == CACHE_SIGNATURE:
            item["semantic"] = "key_cache" if cache_seen == 0 else "value_cache"
            cache_seen += 1
        else:
            item["semantic"] = EXPECTED_INPUTS.get(signature, "unknown")
    return result


def parse_outputs(path: Path) -> list[dict]:
    block = next(value for value in target_blocks(path) if 'op: "NetOutput"' in value)
    result = []
    for match in re.finditer(
        r'key: "\[i\]input(\d+)".*?s: \'(name: "input\d+".*?layout:.*?\\n)\'',
        block,
        re.DOTALL,
    ):
        dtype, shape = descriptor(match.group(2))
        result.append({"index": int(match.group(1)), "dtype": dtype, "shape": list(shape)})
    return sorted(result, key=lambda item: item["index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--eager-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_outputs = json.loads(args.eager_screen.read_text(encoding="utf-8"))["output_specs"]
    inputs = parse_inputs(args.graph)
    outputs = parse_outputs(args.graph)
    for item, expected in zip(outputs, expected_outputs):
        item["semantic"] = expected["name"]
    expected_input_names = {
        "hidden", "position", "sequence_length", "block_table", "slot_mapping",
        "key_cache", "value_cache", "explicit_tiling",
    }
    input_valid = (
        len(inputs) == 8
        and {item["semantic"] for item in inputs} == expected_input_names
        and [item["index"] for item in inputs] == list(range(8))
    )
    output_valid = len(outputs) == len(expected_outputs) and all(
        item["index"] == index
        and item["dtype"] == expected["abi_dtype"]
        and item["shape"] == expected["shape"]
        for index, (item, expected) in enumerate(zip(outputs, expected_outputs))
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
