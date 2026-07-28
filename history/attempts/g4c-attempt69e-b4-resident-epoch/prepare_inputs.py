#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BLOCK_SIZE = 128
BLOCK_TABLE = np.asarray([[1, 0], [3, 2], [5, 4], [7, 6]], dtype=np.int32)
CASES = (
    ("k1-heterogeneous", 1, "case0", [0, 1, 2, 3], [1, 1, 1, 1]),
    ("k2-heterogeneous", 2, "case0", [0, 1, 2, 3], [1, 1, 1, 1]),
    ("k4-heterogeneous", 4, "case0", [0, 1, 2, 3], [1, 1, 1, 1]),
    ("k8-all-active", 8, "case0", [0, 0, 0, 0], [1, 1, 1, 1]),
    ("active-empty-alternating", 4, "case1", [1, 0, 2, 0], [1, 0, 1, 0]),
    (
        "finished-active-empty-active",
        4,
        "case2",
        [3, 1, 0, 2],
        [0, 1, 0, 1],
    ),
    ("independent-early-eos", 4, "case0", [0, 1, 2, 3], [1, 1, 1, 1]),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: np.ndarray, dtype: np.dtype) -> dict:
    np.ascontiguousarray(value, dtype=dtype).tofile(path)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "dtype": str(np.dtype(dtype)),
        "shape": list(value.shape),
    }


def slots(positions: np.ndarray) -> np.ndarray:
    result = []
    for request, position in enumerate(positions):
        block = BLOCK_TABLE[request, int(position) // BLOCK_SIZE]
        result.append(int(block) * BLOCK_SIZE + int(position) % BLOCK_SIZE)
    return np.asarray(result, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "reference_sha256": sha256(args.reference),
        "block_table": BLOCK_TABLE.tolist(),
        "cases": [],
    }
    with np.load(args.reference) as reference:
        tiling = reference["tiling"].copy()
        for name, max_steps, source_case, position_values, active_values in CASES:
            case_dir = args.output_dir / name
            case_dir.mkdir()
            positions = np.asarray(position_values, dtype=np.int64)
            active = np.asarray(active_values, dtype=np.int32)
            if name == "active-empty-alternating":
                tokens = reference["case1_token"].copy()
                lengths = reference["case1_sequence_length"].copy()
            elif name == "finished-active-empty-active":
                tokens = reference["case2_token"].copy()
                lengths = reference["case2_sequence_length"].copy()
            else:
                tokens = reference["case0_token"].copy()
                lengths = (positions + 1).astype(np.int32).reshape(4, 1)
            values = (
                ("token_id.bin", tokens, np.int64),
                ("position.bin", positions, np.int64),
                ("sequence_length.bin", lengths, np.int32),
                (
                    "key_cache.bin",
                    reference[f"{source_case}_input_key_cache_bits"].copy(),
                    np.uint16,
                ),
                ("slot_mapping.bin", slots(positions), np.int32),
                ("active_mask.bin", active, np.int32),
                ("block_table.bin", BLOCK_TABLE, np.int32),
                (
                    "value_cache.bin",
                    reference[f"{source_case}_input_value_cache_bits"].copy(),
                    np.uint16,
                ),
                ("explicit_tiling.bin", tiling, np.uint8),
            )
            files = {
                filename: write(case_dir / filename, value, dtype)
                for filename, value, dtype in values
            }
            manifest["cases"].append(
                {
                    "name": name,
                    "max_steps": max_steps,
                    "source_case": source_case,
                    "position": positions.tolist(),
                    "active_mask": active.tolist(),
                    "files": files,
                }
            )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
