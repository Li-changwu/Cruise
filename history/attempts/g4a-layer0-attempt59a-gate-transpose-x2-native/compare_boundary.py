#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3
BASELINE_OUTPUT_NAMES = (
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
    "gate_preactivation",
    "gate",
    "up",
    "mlp_product",
    "mlp_projection",
    "hidden_after_mlp",
)
CANDIDATE_OUTPUT_NAMES = BASELINE_OUTPUT_NAMES
PREFIX_NAMES = CANDIDATE_OUTPUT_NAMES[: CANDIDATE_OUTPUT_NAMES.index("gate_preactivation")]


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
    names: tuple[str, ...], native: dict[str, np.ndarray], reference: dict[str, np.ndarray], step: int
) -> str | None:
    for name in names:
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

    missing_common_reference_keys = sorted(set(baseline_reference) - set(candidate_reference))
    common_reference_differences = sorted(
        name
        for name in set(baseline_reference) & set(candidate_reference)
        if not np.array_equal(baseline_reference[name], candidate_reference[name])
    )
    common_eager_references_exact = (
        not missing_common_reference_keys and not common_reference_differences
    )

    prefix_differences = []
    candidate_all_within_tolerance = True
    steps = []
    for step in range(1, 5):
        candidate_metrics = {}
        for name in CANDIDATE_OUTPUT_NAMES:
            metric = metrics(
                candidate_native[f"step{step}_{name}"],
                candidate_reference[reference_key(step, name)],
            )
            candidate_metrics[name] = metric
            candidate_all_within_tolerance = (
                candidate_all_within_tolerance and metric["all_within_tolerance"]
            )
        for name in PREFIX_NAMES:
            if not np.array_equal(
                baseline_native[f"step{step}_{name}"],
                candidate_native[f"step{step}_{name}"],
            ):
                prefix_differences.append(f"step{step}_{name}")
        steps.append(
            {
                "step": step,
                "baseline_first_exact_mismatch": first_mismatch(
                    BASELINE_OUTPUT_NAMES, baseline_native, baseline_reference, step
                ),
                "candidate_first_exact_mismatch": first_mismatch(
                    CANDIDATE_OUTPUT_NAMES, candidate_native, candidate_reference, step
                ),
                "candidate_outputs": candidate_metrics,
            }
        )

    native_prefix_exact = not prefix_differences
    step1_preactivation_exact = steps[0]["candidate_outputs"]["gate_preactivation"]["all_exact"]
    step1_gate_exact = steps[0]["candidate_outputs"]["gate"]["all_exact"]
    baseline_step1_first = steps[0]["baseline_first_exact_mismatch"]
    if not common_eager_references_exact or not native_prefix_exact:
        mechanism_result = "confounded"
    elif not step1_preactivation_exact:
        mechanism_result = "transpose_x2_not_exact"
    elif not step1_gate_exact:
        mechanism_result = "silu_not_exact"
    else:
        mechanism_result = "transpose_x2_gate_path_exact"

    diagnostic_valid = (
        common_eager_references_exact
        and native_prefix_exact
        and baseline_step1_first == "gate_preactivation"
        and candidate_all_within_tolerance
        and mechanism_result == "transpose_x2_gate_path_exact"
    )
    result = {
        "gate": "Attempt 59a gate MatMulV2 transpose_x2 probe",
        "diagnostic_valid": diagnostic_valid,
        "mechanism_supported": diagnostic_valid,
        "mechanism_result": mechanism_result,
        "common_eager_references_arraywise_exact": common_eager_references_exact,
        "missing_common_reference_keys": missing_common_reference_keys,
        "common_eager_reference_differences": common_reference_differences,
        "native_prefix_arraywise_exact": native_prefix_exact,
        "native_prefix_differences": prefix_differences,
        "candidate_all_within_tolerance": candidate_all_within_tolerance,
        "step1_gate_preactivation": steps[0]["candidate_outputs"]["gate_preactivation"],
        "step1_gate": steps[0]["candidate_outputs"]["gate"],
        "steps": steps,
        "claim_boundary": (
            "Layer-0 gate MatMulV2 transpose_x2 lowering only; this cannot pass complete G4a."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "diagnostic_valid": diagnostic_valid,
                "mechanism_supported": diagnostic_valid,
                "mechanism_result": mechanism_result,
                "common_eager_references_arraywise_exact": common_eager_references_exact,
                "native_prefix_arraywise_exact": native_prefix_exact,
                "baseline_first_exact_mismatch": [
                    item["baseline_first_exact_mismatch"] for item in steps
                ],
                "candidate_first_exact_mismatch": [
                    item["candidate_first_exact_mismatch"] for item in steps
                ],
                "step1_gate_preactivation": result["step1_gate_preactivation"],
                "step1_gate": result["step1_gate"],
                "candidate_all_within_tolerance": candidate_all_within_tolerance,
            },
            indent=2,
        ),
        flush=True,
    )
    if not diagnostic_valid:
        raise SystemExit(92)


if __name__ == "__main__":
    main()
