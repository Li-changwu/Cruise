#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def content_sha256(values: dict[str, np.ndarray], names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    baseline_names = sorted(baseline)
    missing = sorted(set(baseline) - set(candidate))
    differences = sorted(
        name
        for name in set(baseline) & set(candidate)
        if not np.array_equal(baseline[name], candidate[name])
    )
    boundary_names = (
        "input_norm", "query_rope", "key_rope", "value_projection",
        "masked_scores", "probabilities", "attention_value",
        "attention_projection", "hidden_after_attention", "post_attention_norm",
        "gate_preactivation", "gate", "up", "mlp_product", "mlp_projection",
        "hidden_after_mlp",
    )
    expected_extra = {f"step{step}_layer_hidden_bits" for step in range(1, 5)}
    expected_extra.update(
        f"step{step}_layer0_{name}_bits"
        for step in range(1, 5)
        for name in boundary_names
    )
    extra = set(candidate) - set(baseline)
    baseline_content_sha256 = content_sha256(baseline, baseline_names)
    candidate_content_sha256 = (
        content_sha256(candidate, baseline_names) if not missing else None
    )
    result = {
        "valid": (
            not missing
            and not differences
            and extra == expected_extra
            and baseline_content_sha256 == candidate_content_sha256
        ),
        "baseline_array_count": len(baseline),
        "candidate_array_count": len(candidate),
        "baseline_content_sha256": baseline_content_sha256,
        "candidate_common_content_sha256": candidate_content_sha256,
        "missing_arrays": missing,
        "different_common_arrays": differences,
        "extra_arrays": sorted(extra),
        "expected_extra_arrays": sorted(expected_extra),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["valid"]:
        raise SystemExit(90)


if __name__ == "__main__":
    main()
