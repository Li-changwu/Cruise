#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status_pass(path: Path) -> tuple[bool, list[dict[str, str]]]:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return bool(rows) and all(int(row["exit_status"]) == 0 for row in rows), rows


def metric_pass(metric: dict[str, Any]) -> bool:
    return bool(metric.get("all_within_tolerance"))


def step_pass(step: dict[str, Any]) -> bool:
    finite = step.get("host_finite", step.get("host_logits_finite", True)) and step.get(
        "device_finite", step.get("device_logits_finite", True)
    )
    return bool(
        step.get("pass")
        and metric_pass(step["logits_host_device"])
        and finite
        and step["host_token"] == step["device_token"]
        and step["host_token"] == step["host_argmax"]
        and step["device_token"] == step["device_argmax"]
    )


def b1_pass(result: dict[str, Any]) -> bool:
    steps = result.get("steps", [])
    return bool(
        result.get("pass")
        and result.get("execution_success")
        and result.get("rtol") == 0.005
        and result.get("atol") == 0.005
        and [step.get("step") for step in steps] == [1, 2, 3, 4]
        and all(
            step.get("pass")
            and step.get("logits_finite")
            and step.get("greedy_equal")
            and step.get("next_position") == step.get("expected_next_position")
            and step.get("unaddressed_key_elementwise_exact")
            and step.get("unaddressed_value_elementwise_exact")
            and all(
                metric_pass(step[name])
                for name in (
                    "logits_vs_eager",
                    "key_cache_vs_eager",
                    "value_cache_vs_eager",
                    "written_key_vs_eager",
                    "written_value_vs_eager",
                )
            )
            for step in steps
        )
    )


def b1_epoch_case_pass(case: dict[str, Any]) -> bool:
    return bool(
        case.get("pass")
        and case.get("token_history_equal")
        and case.get("padded_tokens_are_minus_one")
        and case.get("padded_logits_are_zero")
        and metric_pass(case["key_cache_host_device"])
        and metric_pass(case["value_cache_host_device"])
        and case.get("host_unaddressed_key_exact")
        and case.get("device_unaddressed_key_exact")
        and case.get("host_unaddressed_value_exact")
        and case.get("device_unaddressed_value_exact")
        and case.get("position_equal")
        and case.get("control_equal")
        and case.get("host_control_expected")
        and case.get("device_control_expected")
        and case.get("host_submission_count_expected")
        and case.get("device_one_feed_fetch")
        and case["device_runtime"]["host_model_submissions"] == 1
        and case["device_runtime"]["feed_calls"] == 1
        and case["device_runtime"]["fetch_calls"] == 1
        and all(step_pass(step) for step in case.get("steps", []))
    )


def b1_epoch_pass(result: dict[str, Any]) -> bool:
    cases = {case["case"]: case for case in result.get("cases", [])}
    return bool(
        result.get("pass")
        and result.get("controlled_early_eos_pass")
        and set(cases) == {"k1", "k2", "k4", "k8", "early-eos"}
        and result.get("eager_k1_continuity", {}).get("pass")
        and all(b1_epoch_case_pass(case) for case in cases.values())
    )


def fixed_batch_case_pass(case: dict[str, Any]) -> bool:
    frozen = (
        "host_inactive_request_key_exact",
        "device_inactive_request_key_exact",
        "host_inactive_request_value_exact",
        "device_inactive_request_value_exact",
        "inactive_token_length_position_slot_frozen",
    )
    return bool(
        case.get("pass")
        and case.get("token_history_equal")
        and case.get("padded_tokens_are_minus_one")
        and case.get("padded_logits_are_zero")
        and metric_pass(case["key_cache_host_device"])
        and metric_pass(case["value_cache_host_device"])
        and case.get("host_unaddressed_key_exact")
        and case.get("device_unaddressed_key_exact")
        and case.get("host_unaddressed_value_exact")
        and case.get("device_unaddressed_value_exact")
        and all(case.get(name) for name in frozen)
        and all(case.get("final_fields", {}).values())
        and case.get("host_submission_count_expected")
        and case.get("device_one_feed_fetch")
        and case["device_runtime"]["host_model_submissions"] == 1
        and case["device_runtime"]["feed_calls"] == 1
        and case["device_runtime"]["fetch_calls"] == 1
        and all(step_pass(step) for step in case.get("steps", []))
    )


def fixed_batch_pass(
    result: dict[str, Any], expected_cases: set[str]
) -> bool:
    cases = {case["case"]: case for case in result.get("cases", [])}
    continuity = result.get("eager_continuity", {})
    return bool(
        result.get("pass")
        and result.get("independent_eos_pass")
        and set(cases) == expected_cases
        and continuity.get("pass")
        and all(fixed_batch_case_pass(case) for case in cases.values())
    )


def recovery_pass(result: dict[str, Any]) -> bool:
    expected = {
        "invalid-max-steps": 201,
        "capacity-exceeded": 202,
        "unsupported-sampling": 205,
        "unsupported-graph": 206,
    }
    cases = {case["case"]: case for case in result.get("cases", [])}
    return bool(
        result.get("pass")
        and set(cases) == set(expected)
        and all(
            case.get("pass")
            and case.get("expected_status") == expected[name]
            and case.get("observed_status") == expected[name]
            and case.get("control_exact")
            and case.get("model_calls_zero")
            and case.get("fallback_flag_set")
            and case.get("logits_zero")
            and case.get("token_history_minus_one")
            and all(case.get("state_exact", {}).values())
            and case.get("one_feed_fetch")
            for name, case in cases.items()
        )
    )


def performance_pass(result: dict[str, Any]) -> bool:
    cases = {case["k"]: case for case in result.get("cases", [])}
    thresholds = result.get("thresholds", {})
    return bool(
        result.get("pass")
        and result.get("expected_rows") == 126
        and result.get("observed_rows") == 126
        and result.get("block_files_chronological")
        and all(result.get("block_structure", {}).values())
        and set(cases) == {2, 4, 8}
        and thresholds.get("measured_pairs_per_k") == 15
        and thresholds.get("minimum_median_speedup") == 1.10
        and thresholds.get("minimum_device_faster_pairs") == 13
        and all(
            case.get("pass")
            and case.get("warmup_blocks_valid")
            and case.get("measured_pairs") == 15
            and case.get("abba_order")
            and case.get("submission_semantics")
            and case.get("paired_speedup_median", 0) >= 1.10
            and case.get("device_faster_pairs", 0) >= 13
            and case.get("iqr_separated")
            and case["device_wall_us"]["median"] < case["host_wall_us"]["median"]
            and case["device_cpu_us"]["median"] < case["host_cpu_us"]["median"]
            for case in cases.values()
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("g4a", "g4b", "b2", "b4", "recovery", "performance"):
        parser.add_argument(f"--{name}-result", required=True, type=Path)
        parser.add_argument(f"--{name}-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        name: {
            "result": getattr(args, f"{name}_result"),
            "status": getattr(args, f"{name}_status"),
        }
        for name in ("g4a", "g4b", "b2", "b4", "recovery", "performance")
    }
    results = {name: load_json(item["result"]) for name, item in paths.items()}
    status_checks = {name: status_pass(item["status"])[0] for name, item in paths.items()}
    semantic_checks = {
        "g4a_fixed_b1_complete_decoder": b1_pass(results["g4a"]),
        "g4b_b1_resident_epoch": b1_epoch_pass(results["g4b"]),
        "g4c_fixed_b2": fixed_batch_pass(
            results["b2"],
            {
                "k1-heterogeneous",
                "k2-heterogeneous",
                "k4-heterogeneous",
                "k8-both-active",
                "active-empty",
                "finished-active",
                "independent-early-eos",
            },
        ),
        "g4c_fixed_b4": fixed_batch_pass(
            results["b4"],
            {
                "k1-heterogeneous",
                "k2-heterogeneous",
                "k4-heterogeneous",
                "k8-all-active",
                "active-empty-alternating",
                "finished-active-empty-active",
                "independent-early-eos",
            },
        ),
        "safe_preexecution_fallback": recovery_pass(results["recovery"]),
        "stable_k_ge_2_performance": performance_pass(results["performance"]),
    }
    overall = all(status_checks.values()) and all(semantic_checks.values())
    output = {
        "gate": "G4 device-resident fixed-batch generation kernel completion",
        "pass": overall,
        "semantic_checks": semantic_checks,
        "all_status_rows_zero": status_checks,
        "input_sha256": {
            name: {
                key: sha256(path) for key, path in item.items()
            }
            for name, item in paths.items()
        },
        "scope_boundary": (
            "Passing closes fixed B=1/B=2/B=4 decoder-epoch correctness, "
            "one-Feed/one-Fetch control, K>=2 stable performance, and frozen "
            "pre-execution fallback on one 910B2/CANN 9.0.0 server. It does "
            "not include vLLM scheduling, request insertion, preemption, or "
            "continuous batching."
        ),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(json.dumps(output, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
