#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    records = []
    for name in shards:
        path = args.snapshot / name
        resolved = path.resolve(strict=True)
        expected_sha = resolved.name
        actual_sha = file_sha256(resolved)
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        records.append({
            "name": name,
            "size_bytes": resolved.stat().st_size,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "sha256_valid": actual_sha == expected_sha,
            "tensor_count": len(keys),
        })
    mapped = set(index["weight_map"])
    observed = set()
    for name in shards:
        with safe_open(args.snapshot / name, framework="pt", device="cpu") as handle:
            observed.update(handle.keys())
    result = {
        "snapshot": str(args.snapshot),
        "shard_count": len(shards),
        "mapped_tensor_count": len(mapped),
        "observed_tensor_count": len(observed),
        "all_shards_sha256_valid": all(item["sha256_valid"] for item in records),
        "tensor_index_exact": observed == mapped,
        "shards": records,
    }
    result["valid"] = result["all_shards_sha256_valid"] and result["tensor_index_exact"]
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

