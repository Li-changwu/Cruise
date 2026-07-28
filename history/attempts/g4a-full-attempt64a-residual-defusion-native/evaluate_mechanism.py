#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_exact = [
        item["first_layer_hidden_exact_mismatch"] for item in baseline["steps"]
    ]
    baseline_tolerance = [
        item["first_layer_hidden_tolerance_failure"] for item in baseline["steps"]
    ]
    candidate_exact = [
        item["first_layer_hidden_exact_mismatch"] for item in candidate["steps"]
    ]
    candidate_tolerance = [
        item["first_layer_hidden_tolerance_failure"] for item in candidate["steps"]
    ]
    all_layer_hiddens_exact = all(value is None for value in candidate_exact)
    all_layer_hiddens_within_tolerance = all(
        value is None for value in candidate_tolerance
    )
    baseline_valid = (
        baseline.get("diagnostic_complete") is True
        and baseline_exact == [0, 0, 0, 0]
        and baseline_tolerance == [1, 1, 1, 1]
    )
    supported = (
        baseline_valid
        and candidate.get("diagnostic_complete") is True
        and candidate.get("pass") is True
        and all_layer_hiddens_within_tolerance
        and all(value is None or value > 0 for value in candidate_exact)
    )
    result = {
        "supported": supported,
        "baseline_valid": baseline_valid,
        "candidate_diagnostic_complete": candidate.get("diagnostic_complete"),
        "candidate_g4a_outputs_pass": candidate.get("pass"),
        "all_layer_hiddens_exact": all_layer_hiddens_exact,
        "all_layer_hiddens_within_tolerance": all_layer_hiddens_within_tolerance,
        "baseline_first_layer_hidden_exact_mismatch": baseline_exact,
        "baseline_first_layer_hidden_tolerance_failure": baseline_tolerance,
        "candidate_first_layer_hidden_exact_mismatch": candidate_exact,
        "candidate_first_layer_hidden_tolerance_failure": candidate_tolerance,
        "baseline_sha256": sha256(args.baseline),
        "candidate_sha256": sha256(args.candidate),
        "claim_boundary": (
            "Passing supports residual defusion in the 5-output diagnostic graph; "
            "a clean graph without layer-hidden outputs is still required for G4a."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not supported:
        raise SystemExit(90)


if __name__ == "__main__":
    main()
