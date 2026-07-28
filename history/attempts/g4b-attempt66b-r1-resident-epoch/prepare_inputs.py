#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value)
    value.tofile(path)
    return {
        "name": path.name,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(args.reference) as reference:
        values = {
            "token_id.bin": reference["token_ids"][:1].copy().reshape(1, 1),
            "position.bin": np.asarray([0], dtype=np.int64),
            "sequence_length.bin": np.asarray([[1]], dtype=np.int32),
            "key_cache.bin": reference["input_key_cache_bits"].copy(),
            "slot_mapping.bin": np.asarray([128], dtype=np.int32),
            "block_table.bin": reference["block_table"].copy(),
            "value_cache.bin": reference["input_value_cache_bits"].copy(),
            "explicit_tiling.bin": reference["tiling"].copy(),
        }
    manifest = {
        "reference_sha256": sha256(args.reference),
        "air_abi_order": list(values),
        "files": [write(args.output_dir / name, value) for name, value in values.items()],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
