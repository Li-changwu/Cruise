#!/usr/bin/env python3
import argparse
import json

import numpy as np


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    args = parser.parse_args()

    with np.load(args.reference) as reference:
        hf = bf16_to_float32(reference["step4_hf_probabilities_bits"])
        fixed = bf16_to_float32(reference["step4_manual_probabilities_bits"])
        active = bf16_to_float32(reference["step4_active_probabilities_bits"])
        scores = reference["step4_manual_scaled_scores"][..., :4]

    exact_indices = np.argwhere(fixed != hf)
    tolerance_indices = np.argwhere(~np.isclose(fixed, hf, rtol=5e-3, atol=5e-3))
    records = []
    for batch, head, query, token in exact_indices:
        records.append(
            {
                "head": int(head),
                "token": int(token),
                "scaled_score": float(scores[batch, head, query, token]),
                "hf": float(hf[batch, head, query, token]),
                "fixed_capacity": float(fixed[batch, head, query, token]),
                "active_length": float(active[batch, head, query, token]),
                "within_tolerance": bool(
                    np.isclose(
                        fixed[batch, head, query, token],
                        hf[batch, head, query, token],
                        rtol=5e-3,
                        atol=5e-3,
                    )
                ),
            }
        )

    result = {
        "exact_mismatch_count": int(exact_indices.shape[0]),
        "tolerance_failure_count": int(tolerance_indices.shape[0]),
        "active_matches_hf_exactly": bool(np.array_equal(active, hf)),
        "records": records,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
