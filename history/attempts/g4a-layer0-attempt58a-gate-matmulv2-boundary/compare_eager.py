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


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    missing = sorted(set(baseline) - set(candidate))
    extra = sorted(set(candidate) - set(baseline))
    differences = sorted(
        name
        for name in set(baseline) & set(candidate)
        if not np.array_equal(baseline[name], candidate[name])
    )
    result = {
        "valid": not missing and not extra and not differences,
        "baseline_sha256": sha256(args.baseline),
        "candidate_sha256": sha256(args.candidate),
        "array_count": len(candidate),
        "missing_arrays": missing,
        "extra_arrays": extra,
        "different_arrays": differences,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["valid"]:
        raise SystemExit(90)


if __name__ == "__main__":
    main()
