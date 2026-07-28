#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCALED_SHAPE = (1, 28, 1, 8)
RAW_SHAPE = (1, 28, 8)
CANDIDATES = ("legacy_scaled", "fp32_div_bf16", "fp32_mul_bf16")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16_file_to_float32(path: Path) -> np.ndarray:
    bits = np.fromfile(path, dtype="<u2").reshape(RAW_SHAPE)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
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
        raw = bf16_file_to_float32(args.native_dir / f"step{step}_raw_qk.bin")
        raw_ref = direct[f"step{step}_custom_raw"].astype(np.float32)
        item = {"step": step, "position": step - 1, "raw_vs_direct": metrics(raw, raw_ref)}
        scaled_ref = eager[f"step{step}_qk_scores"].astype(np.float32)
        for name in CANDIDATES:
            actual = np.fromfile(args.native_dir / f"step{step}_{name}.bin", dtype="<f4").reshape(SCALED_SHAPE)
            item[name + "_vs_eager"] = metrics(actual, scaled_ref)
        steps.append(item)

    raw_all_exact = all(step["raw_vs_direct"]["all_exact"] for step in steps)
    candidate_all_exact = {
        name: all(step[name + "_vs_eager"]["all_exact"] for step in steps)
        for name in CANDIDATES
    }
    repaired = [name for name, passed in candidate_all_exact.items() if name != "legacy_scaled" and passed]
    result = {
        "gate": "G4a QK Attempt 50 raw boundary and scaling semantics",
        "execution_success": True,
        "raw_all_positions_exact": raw_all_exact,
        "candidate_all_positions_exact": candidate_all_exact,
        "repair_candidates": repaired,
        "pass": raw_all_exact and bool(repaired),
        "eager_reference_sha256": sha256(args.eager_reference),
        "direct_reference_sha256": sha256(args.direct_reference),
        "steps": steps,
        "claim_boundary": "Frozen QK boundary only; full decoder and device-resident generation remain untested.",
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("G4A_QK_ATTEMPT50_COMPARE " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

