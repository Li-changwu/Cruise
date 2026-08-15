#!/usr/bin/env python3
"""Render public GraphPp configs and the independent KV output oracle."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


BATCH_SIZE = 4
KV_HEADS = 4
HEAD_DIM = 128
BLOCK_SIZE = 128
BLOCKS_PER_ROW = 3
PHYSICAL_BLOCKS = BATCH_SIZE * BLOCKS_PER_ROW
PACK_WIDTH = 16
CACHE_SHAPE = (PHYSICAL_BLOCKS, KV_HEADS * HEAD_DIM // PACK_WIDTH, BLOCK_SIZE, PACK_WIDTH)
CACHE_ELEMENTS = PHYSICAL_BLOCKS * (KV_HEADS * HEAD_DIM // PACK_WIDTH) * BLOCK_SIZE * PACK_WIDTH
SLOTS = tuple(row * BLOCKS_PER_ROW * BLOCK_SIZE for row in range(BATCH_SIZE))
KEY_BITS = (0x3F80, 0x4000, 0x4040, 0x4080)
VALUE_BITS = (0xBF80, 0xC000, 0xC040, 0xC080)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expected_cache(bits: tuple[int, ...]) -> bytearray:
    output = bytearray(CACHE_ELEMENTS * 2)
    packed_heads = KV_HEADS * HEAD_DIM // PACK_WIDTH
    for row, slot in enumerate(SLOTS):
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        for packed_head in range(packed_heads):
            for lane in range(PACK_WIDTH):
                index = (((block * packed_heads + packed_head) * BLOCK_SIZE + offset) * PACK_WIDTH + lane)
                struct.pack_into("<H", output, index * 2, bits[row])
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--deploy-output", type=Path, required=True)
    parser.add_argument("--expected-output", type=Path, required=True)
    args = parser.parse_args()

    write_json(
        args.graph_output,
        {
            "inputs_tensor_desc": [
                {"data_type": "DT_BFLOAT16", "shape": [4, 4, 128]},
                {"data_type": "DT_BFLOAT16", "shape": [4, 4, 128]},
                {"data_type": "DT_BFLOAT16", "shape": list(CACHE_SHAPE)},
                {"data_type": "DT_BFLOAT16", "shape": list(CACHE_SHAPE)},
                {"data_type": "DT_INT32", "shape": [4]},
            ]
        },
    )
    write_json(
        args.deploy_output,
        {
            "batch_deploy_info": [
                {
                    "flow_node_list": ["kv_alias_node"],
                    "logic_device_list": "0:0:0:0",
                }
            ]
        },
    )
    args.expected_output.parent.mkdir(parents=True, exist_ok=True)
    args.expected_output.write_bytes(expected_cache(KEY_BITS) + expected_cache(VALUE_BITS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
