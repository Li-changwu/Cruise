#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3
CACHE_SHAPE = (28, 2, 128, 4, 128)
LOGITS_SHAPE = (1, 1, 152064)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual_fp32 = actual.astype(np.float32)
    expected_fp32 = expected.astype(np.float32)
    exact = np.equal(actual, expected)
    close = np.isclose(actual_fp32, expected_fp32, rtol=RTOL, atol=ATOL)
    return {
        "all_exact": bool(np.all(exact)),
        "all_within_tolerance": bool(np.all(close)),
        "max_abs_error": float(np.max(np.abs(actual_fp32 - expected_fp32))),
        "exact_mismatch_count": int(exact.size - np.count_nonzero(exact)),
        "tolerance_failure_count": int(close.size - np.count_nonzero(close)),
    }


def bf16_metrics(actual_bits: np.ndarray, expected_bits: np.ndarray) -> dict:
    actual = (actual_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    expected = (expected_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return metrics(actual, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logits = np.fromfile(args.actual_dir / "logits.bin", dtype=np.float32).reshape(
        LOGITS_SHAPE
    )
    key_bits = np.fromfile(args.actual_dir / "key_cache.bin", dtype=np.uint16).reshape(
        CACHE_SHAPE
    )
    value_bits = np.fromfile(
        args.actual_dir / "value_cache.bin", dtype=np.uint16
    ).reshape(CACHE_SHAPE)
    next_position = np.fromfile(
        args.actual_dir / "next_position.bin", dtype=np.int64
    ).reshape(1)
    with np.load(args.reference) as reference:
        result = {
            "gate": "G4b Attempt 66a-r5 all_ops C++ DataFlow full-decoder smoke",
            "pass": False,
            "reference_sha256": sha256(args.reference),
            "feed_calls": 1,
            "fetch_calls": 1,
            "logits_vs_eager": metrics(logits, reference["step1_logits"]),
            "key_cache_vs_eager": bf16_metrics(
                key_bits, reference["step1_key_cache_bits"]
            ),
            "value_cache_vs_eager": bf16_metrics(
                value_bits, reference["step1_value_cache_bits"]
            ),
            "position_equal": bool(
                np.array_equal(next_position, reference["step1_next_position"])
            ),
            "claim_boundary": (
                "One C++ Host DataFlow Feed/Fetch only; Device UDF recurrence, "
                "sampling and EOS are not tested."
            ),
        }
    result["pass"] = bool(
        result["logits_vs_eager"]["all_within_tolerance"]
        and result["key_cache_vs_eager"]["all_within_tolerance"]
        and result["value_cache_vs_eager"]["all_within_tolerance"]
        and result["position_equal"]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
