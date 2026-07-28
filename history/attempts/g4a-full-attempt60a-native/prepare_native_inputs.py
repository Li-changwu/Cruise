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


def write_array(path: Path, value: np.ndarray, dtype: np.dtype) -> None:
    np.ascontiguousarray(value, dtype=dtype).tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"native input directory is not empty: {args.output_dir}")

    with np.load(args.reference) as reference:
        token_ids = reference["token_ids"]
        if token_ids.shape != (4,):
            raise RuntimeError(f"unexpected token shape: {token_ids.shape}")
        write_array(
            args.output_dir / "initial_key_cache.bin",
            reference["input_key_cache_bits"],
            np.uint16,
        )
        write_array(
            args.output_dir / "initial_value_cache.bin",
            reference["input_value_cache_bits"],
            np.uint16,
        )
        write_array(args.output_dir / "block_table.bin", reference["block_table"], np.int32)
        write_array(args.output_dir / "tiling.bin", reference["tiling"], np.uint8)
        for step, token_id in enumerate(token_ids, start=1):
            write_array(
                args.output_dir / f"step{step}_token_id.bin",
                np.asarray([[token_id]], dtype=np.int64),
                np.int64,
            )

    files = sorted(args.output_dir.iterdir())
    manifest = {
        "reference_sha256": sha256(args.reference),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
