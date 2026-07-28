#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3
OUTPUT_NAMES = (
    "updated_key",
    "updated_value",
    "next_position",
    "input_norm",
    "query_rope",
    "key_rope",
    "value_projection",
    "masked_scores",
    "probabilities",
    "attention_value",
    "attention_projection",
    "hidden_after_attention",
    "post_attention_norm",
    "gate",
    "up",
    "mlp_product",
    "mlp_projection",
    "hidden_after_mlp",
)
PREFIX_NAMES = OUTPUT_NAMES[: OUTPUT_NAMES.index("gate")]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def metrics(native: np.ndarray, expected: np.ndarray) -> dict:
    exact = native == expected
    if native.dtype == np.uint16 and expected.dtype == np.uint16:
        actual_fp32 = bf16_to_float32(native)
        expected_fp32 = bf16_to_float32(expected)
    else:
        actual_fp32 = native.astype(np.float32)
        expected_fp32 = expected.astype(np.float32)
    close = np.isclose(actual_fp32, expected_fp32, rtol=RTOL, atol=ATOL)
    error = np.abs(actual_fp32 - expected_fp32)
    return {
        "all_exact": bool(np.all(exact)),
        "exact_mismatch_count": int(exact.size - np.count_nonzero(exact)),
        "all_within_tolerance": bool(np.all(close)),
        "tolerance_failure_count": int(close.size - np.count_nonzero(close)),
        "max_abs_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error)),
    }


def reference_key(step: int, name: str) -> str:
    suffix = "" if name == "next_position" else "_bits"
    return f"step{step}_{name}{suffix}"


def first_mismatch(
    native: dict[str, np.ndarray], reference: dict[str, np.ndarray], step: int
) -> str | None:
    for name in OUTPUT_NAMES:
        if not np.array_equal(
            native[f"step{step}_{name}"], reference[reference_key(step, name)]
        ):
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-native", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path, required=True)
    parser.add_argument("--candidate-native", type=Path, required=True)
    parser.add_argument("--candidate-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_native = load_npz(args.baseline_native)
    baseline_reference = load_npz(args.baseline_reference)
    candidate_native = load_npz(args.candidate_native)
    candidate_reference = load_npz(args.candidate_reference)

    reference_keys_match = set(baseline_reference) == set(candidate_reference)
    reference_differences = sorted(
        name
        for name in set(baseline_reference) & set(candidate_reference)
        if not np.array_equal(baseline_reference[name], candidate_reference[name])
    )
    eager_references_exact = reference_keys_match and not reference_differences

    prefix_differences = []
    steps = []
    candidate_all_within_tolerance = True
    for step in range(1, 5):
        output_metrics = {}
        for name in OUTPUT_NAMES:
            baseline_key = f"step{step}_{name}"
            expected_key = reference_key(step, name)
            baseline_metric = metrics(
                baseline_native[baseline_key], baseline_reference[expected_key]
            )
            candidate_metric = metrics(
                candidate_native[baseline_key], candidate_reference[expected_key]
            )
            output_metrics[name] = {
                "baseline": baseline_metric,
                "candidate": candidate_metric,
                "exact_mismatch_delta": (
                    candidate_metric["exact_mismatch_count"]
                    - baseline_metric["exact_mismatch_count"]
                ),
                "max_abs_error_delta": (
                    candidate_metric["max_abs_error"] - baseline_metric["max_abs_error"]
                ),
            }
            candidate_all_within_tolerance = (
                candidate_all_within_tolerance
                and candidate_metric["all_within_tolerance"]
            )
            if name in PREFIX_NAMES and not np.array_equal(
                baseline_native[baseline_key], candidate_native[baseline_key]
            ):
                prefix_differences.append(f"step{step}_{name}")
        steps.append(
            {
                "step": step,
                "baseline_first_exact_mismatch": first_mismatch(
                    baseline_native, baseline_reference, step
                ),
                "candidate_first_exact_mismatch": first_mismatch(
                    candidate_native, candidate_reference, step
                ),
                "outputs": output_metrics,
            }
        )

    baseline_step1_first = steps[0]["baseline_first_exact_mismatch"]
    candidate_step1_first = steps[0]["candidate_first_exact_mismatch"]
    candidate_step1_index = (
        len(OUTPUT_NAMES)
        if candidate_step1_first is None
        else OUTPUT_NAMES.index(candidate_step1_first)
    )
    gate_index = OUTPUT_NAMES.index("gate")
    baseline_step1_gate_mismatches = steps[0]["outputs"]["gate"]["baseline"][
        "exact_mismatch_count"
    ]
    candidate_step1_gate_mismatches = steps[0]["outputs"]["gate"]["candidate"][
        "exact_mismatch_count"
    ]
    prefix_native_exact = not prefix_differences
    mechanism_supported = (
        eager_references_exact
        and prefix_native_exact
        and baseline_step1_first == "gate"
        and baseline_step1_gate_mismatches > 0
        and candidate_step1_gate_mismatches == 0
        and candidate_step1_index > gate_index
        and candidate_all_within_tolerance
    )
    if not eager_references_exact or not prefix_native_exact:
        verdict = "confounded"
    elif mechanism_supported:
        verdict = "supported"
    else:
        verdict = "rejected"

    result = {
        "gate": "Attempt 55d versus 55a materialization differential",
        "verdict": verdict,
        "mechanism_supported": mechanism_supported,
        "eager_references_arraywise_exact": eager_references_exact,
        "eager_reference_key_sets_match": reference_keys_match,
        "eager_reference_differences": reference_differences,
        "native_prefix_arraywise_exact": prefix_native_exact,
        "native_prefix_differences": prefix_differences,
        "candidate_all_within_tolerance": candidate_all_within_tolerance,
        "baseline_step1_gate_mismatches": baseline_step1_gate_mismatches,
        "candidate_step1_gate_mismatches": candidate_step1_gate_mismatches,
        "steps": steps,
        "claim_boundary": (
            "Layer-0 gate/up materialization only; this cannot pass complete G4a."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "eager_references_arraywise_exact": eager_references_exact,
                "native_prefix_arraywise_exact": prefix_native_exact,
                "baseline_first_exact_mismatch": [
                    item["baseline_first_exact_mismatch"] for item in steps
                ],
                "candidate_first_exact_mismatch": [
                    item["candidate_first_exact_mismatch"] for item in steps
                ],
                "baseline_step1_gate_mismatches": baseline_step1_gate_mismatches,
                "candidate_step1_gate_mismatches": candidate_step1_gate_mismatches,
                "candidate_all_within_tolerance": candidate_all_within_tolerance,
            },
            indent=2,
        ),
        flush=True,
    )
    if not mechanism_supported:
        raise SystemExit(92)


if __name__ == "__main__":
    main()
