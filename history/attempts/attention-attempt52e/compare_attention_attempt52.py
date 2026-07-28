#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3
NAMES = (
    "attention", "key_cache", "value_cache", "position", "k_projection",
    "k_rope", "qk_scores", "q_projection", "q_rope", "rope_cos", "rope_sin",
    "q_projection_bf16", "k_projection_bf16", "q_rope_bf16", "k_rope_bf16",
    "updated_key_bf16", "qk_scores_bf16",
)
PRIMARY_EXACT = (
    "attention", "key_cache", "value_cache", "position", "qk_scores", "qk_scores_bf16"
)
SHAPES = {
    "attention": (1, 1, 3584), "key_cache": (1, 4, 8, 128),
    "value_cache": (1, 4, 8, 128), "position": (1,),
    "k_projection": (1, 1, 512), "k_rope": (1, 4, 1, 128),
    "qk_scores": (1, 28, 1, 8), "q_projection": (1, 1, 3584),
    "q_rope": (1, 28, 1, 128), "rope_cos": (1, 1, 1, 128),
    "rope_sin": (1, 1, 1, 128), "q_projection_bf16": (1, 1, 3584),
    "k_projection_bf16": (1, 1, 512), "q_rope_bf16": (1, 28, 1, 128),
    "k_rope_bf16": (1, 4, 1, 128), "updated_key_bf16": (1, 4, 8, 128),
    "qk_scores_bf16": (1, 28, 1, 8),
}
BF16_NAMES = set(NAMES[11:])
DTYPES = {name: np.uint16 if name in BF16_NAMES else np.float16 for name in NAMES}
DTYPES["position"] = np.int64
DTYPES["qk_scores"] = np.float32


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(name: str, value: np.ndarray) -> np.ndarray:
    if name in BF16_NAMES:
        return (value.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return value


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    exact = np.equal(actual, expected)
    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    close = np.isclose(actual.astype(np.float32), expected.astype(np.float32), rtol=RTOL, atol=ATOL)
    return {
        "all_exact": bool(np.all(exact)),
        "all_within_tolerance": bool(np.all(close)),
        "max_abs_error": float(np.max(difference)),
        "exact_mismatch_count": int(exact.size - np.count_nonzero(exact)),
        "tolerance_failure_count": int(close.size - np.count_nonzero(close)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--candidate-reference", type=Path, required=True)
    parser.add_argument("--frozen-reference", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = np.load(args.candidate_reference)
    frozen = np.load(args.frozen_reference)
    steps, saved = [], {}
    for step in range(1, 5):
        item = {"step": step, "candidate_vs_frozen": {}, "native_vs_candidate": {}}
        for name in NAMES:
            raw = np.fromfile(args.native_dir / f"step{step}_{name}.bin", dtype=DTYPES[name]).reshape(SHAPES[name])
            value = observed(name, raw)
            saved[f"step{step}_{name}"] = raw
            item["native_vs_candidate"][name] = metrics(value, candidate[f"step{step}_{name}"])
            if name != "qk_scores_bf16":
                item["candidate_vs_frozen"][name] = metrics(
                    candidate[f"step{step}_{name}"], frozen[f"step{step}_{name}"]
                )
        steps.append(item)
    np.savez(args.output_npz, **saved)
    candidate_exact = all(
        value["all_exact"] for step in steps for value in step["candidate_vs_frozen"].values()
    )
    all_within_tolerance = all(
        value["all_within_tolerance"] for step in steps for value in step["native_vs_candidate"].values()
    )
    primary_exact = all(
        step["native_vs_candidate"][name]["all_exact"]
        for step in steps for name in PRIMARY_EXACT
    )
    result = {
        "gate": "G4a Attempt 52e QK-and-softmax opaque-BF16-barrier Attention/KV AIR",
        "execution_success": True,
        "rtol": RTOL,
        "atol": ATOL,
        "candidate_first_sixteen_exact_to_frozen": candidate_exact,
        "native_all_seventeen_within_tolerance": all_within_tolerance,
        "native_primary_qk_attention_kv_position_elementwise_exact": primary_exact,
        "pass": candidate_exact and all_within_tolerance and primary_exact,
        "candidate_reference_sha256": sha256(args.candidate_reference),
        "frozen_reference_sha256": sha256(args.frozen_reference),
        "steps": steps,
        "claim_boundary": "Full layer-0 Attention/KV slice only; not a complete decoder step.",
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("G4A_ATTEMPT52_COMPARE " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
