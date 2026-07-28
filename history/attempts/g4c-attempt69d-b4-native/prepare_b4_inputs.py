#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CASES = (
    "all-active-heterogeneous",
    "two-active-two-empty",
    "finished-active-empty-active",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: np.ndarray, dtype: np.dtype) -> dict:
    np.ascontiguousarray(value, dtype=dtype).tofile(path)
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"reference_sha256": sha256(args.reference), "cases": {}}
    with np.load(args.reference) as reference:
        tiling = reference["tiling"].copy()
        for index, name in enumerate(CASES):
            prefix = f"case{index}"
            case_dir = args.output_dir / prefix
            case_dir.mkdir()
            values = (
                ("token_id.bin", reference[f"{prefix}_token"], np.int64),
                ("position.bin", reference[f"{prefix}_position"], np.int64),
                (
                    "sequence_length.bin",
                    reference[f"{prefix}_sequence_length"],
                    np.int32,
                ),
                (
                    "key_cache.bin",
                    reference[f"{prefix}_input_key_cache_bits"],
                    np.uint16,
                ),
                (
                    "slot_mapping.bin",
                    reference[f"{prefix}_slot_mapping"],
                    np.int32,
                ),
                (
                    "active_mask.bin",
                    reference[f"{prefix}_active_mask"],
                    np.int32,
                ),
                (
                    "block_table.bin",
                    reference[f"{prefix}_block_table"],
                    np.int32,
                ),
                (
                    "value_cache.bin",
                    reference[f"{prefix}_input_value_cache_bits"],
                    np.uint16,
                ),
                ("explicit_tiling.bin", tiling, np.uint8),
            )
            files = {}
            for filename, value, dtype in values:
                files[filename] = write(case_dir / filename, value, dtype)
            manifest["cases"][prefix] = {"semantic_case": name, "files": files}
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
