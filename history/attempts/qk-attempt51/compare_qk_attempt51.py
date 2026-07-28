#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SHAPE = (1, 28, 1, 8)
NAMES = ("legacy_bf16", "fp32_div_bf16", "fp32_mul_bf16")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.fromfile(path, dtype="<u2").reshape(shape)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    diff = np.abs(actual - expected)
    exact = np.equal(actual, expected)
    return {
        "all_exact": bool(np.all(exact)),
        "max_abs_error": float(np.max(diff)),
        "mismatch_count": int(exact.size - np.count_nonzero(exact)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--eager-reference", type=Path, required=True)
    parser.add_argument("--direct-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    eager = np.load(args.eager_reference)
    direct = np.load(args.direct_reference)
    steps = []
    for step in range(1, 5):
        raw = load_bf16(args.native_dir / f"step{step}_raw_qk.bin", (1, 28, 8))
        item = {
            "step": step,
            "raw_vs_direct": metrics(raw, direct[f"step{step}_custom_raw"].astype(np.float32)),
        }
        expected = eager[f"step{step}_qk_scores"].astype(np.float32)
        for name in NAMES:
            item[name + "_vs_eager"] = metrics(
                load_bf16(args.native_dir / f"step{step}_{name}.bin", SHAPE), expected
            )
        steps.append(item)
    raw_exact = all(item["raw_vs_direct"]["all_exact"] for item in steps)
    candidate_exact = {
        name: all(item[name + "_vs_eager"]["all_exact"] for item in steps) for name in NAMES
    }
    repairs = [name for name in NAMES if name != "legacy_bf16" and candidate_exact[name]]
    result = {
        "gate": "G4a QK Attempt 51 materialized BF16 scaling boundary",
        "execution_success": True,
        "raw_all_positions_exact": raw_exact,
        "candidate_all_positions_exact": candidate_exact,
        "repair_candidates": repairs,
        "pass": raw_exact and bool(repairs),
        "eager_reference_sha256": sha256(args.eager_reference),
        "direct_reference_sha256": sha256(args.direct_reference),
        "steps": steps,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("G4A_QK_ATTEMPT51_COMPARE " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

