#!/usr/bin/env python3
"""Independently reconstruct the M4a decision from bounded raw JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MODES = ("eager", "graph", "cruise")
BLOCKED_ORDER = (
    "eager-1",
    "graph-1",
    "cruise-1",
    "cruise-2",
    "graph-2",
    "eager-2",
    "graph-3",
    "cruise-3",
    "eager-3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("missing M4a metric samples")
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _scenario(result: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        item for item in result.get("scenarios", []) if item["scenario"]["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"{result.get('run_label')}: missing scenario {name}")
    return matches[0]


def _semantics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        item["scenario"]["name"]: [
            (
                record["request_index"],
                record.get("tokens"),
                record.get("finish_reason"),
                record.get("stop_reason"),
                record.get("done"),
            )
            for record in item.get("records", [])
        ]
        for item in result.get("scenarios", [])
    }


def _primary_metrics(results: list[dict[str, Any]], scenario_name: str) -> dict[str, float]:
    tpot: list[float] = []
    cpu_seconds = 0.0
    output_tokens = 0
    for result in results:
        scenario = _scenario(result, scenario_name)
        for record in scenario["records"]:
            if record.get("tpot_ms") is not None:
                tpot.append(float(record["tpot_ms"]))
            output_tokens += len(record.get("tokens", []))
        before = float(scenario["process_tree_before"]["cpu_seconds"])
        after = float(scenario["process_tree_after"]["cpu_seconds"])
        cpu_seconds += max(0.0, after - before)
    if output_tokens == 0:
        raise ValueError("primary scenario produced no tokens")
    return {
        "tpot_p50": _percentile(tpot, 50.0),
        "tpot_p95": _percentile(tpot, 95.0),
        "host_cpu_ms_per_output_token": cpu_seconds * 1000.0 / output_tokens,
    }


def _improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline metric must be positive")
    return (baseline - candidate) / baseline * 100.0


def verify(
    comparison_path: Path, workload_path: Path, result_paths: list[Path]
) -> dict[str, Any]:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    labels = [result.get("run_label") for result in results]
    grouped = {
        mode: [result for result in results if result.get("mode") == mode]
        for mode in MODES
    }

    canonical = _semantics(results[0]) if results else {}
    exact_semantics = bool(results) and all(
        _semantics(result) == canonical for result in results
    )
    primary = workload.get("primary_scenario")
    reconstructed = {
        mode: _primary_metrics(grouped[mode], primary)
        for mode in MODES
        if len(grouped[mode]) == 3
    }
    baseline_mode = None
    threshold_values: dict[str, float | str] = {}
    if len(reconstructed) == 3:
        baseline_mode = min(
            ("graph", "eager"),
            key=lambda mode: (
                reconstructed[mode]["tpot_p50"],
                0 if mode == "graph" else 1,
            ),
        )
        baseline = reconstructed[baseline_mode]
        cruise = reconstructed["cruise"]
        threshold_values = {
            "strongest_baseline": baseline_mode,
            "median_tpot_improvement_percent": _improvement(
                baseline["tpot_p50"], cruise["tpot_p50"]
            ),
            "p95_tpot_improvement_percent": _improvement(
                baseline["tpot_p95"], cruise["tpot_p95"]
            ),
            "host_cpu_per_token_reduction_percent": _improvement(
                baseline["host_cpu_ms_per_output_token"],
                cruise["host_cpu_ms_per_output_token"],
            ),
        }

    expected_device_tokens = sum(
        (scenario["max_tokens"] - 1) * scenario["request_count"]
        for scenario in (*workload.get("warmups", []), *workload.get("scenarios", []))
    )
    route_coverage = all(
        result.get("resident_route_metrics", {})
        .get("counters", {})
        .get("device_request_tokens")
        == expected_device_tokens
        for result in grouped["cruise"]
    )
    input_hashes_match = comparison.get("input_sha256") == {
        str(path): _sha256(path) for path in result_paths
    }
    reported_values = comparison.get("threshold_values", {})
    values_match = reported_values.get("strongest_baseline") == baseline_mode and all(
        math.isclose(
            float(reported_values.get(key, float("nan"))),
            float(threshold_values.get(key, float("nan"))),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for key in (
            "median_tpot_improvement_percent",
            "p95_tpot_improvement_percent",
            "host_cpu_per_token_reduction_percent",
        )
    )
    execution_checks = {
        "nine_ordered_inputs": labels == list(BLOCKED_ORDER),
        "three_starts_per_mode": all(len(grouped[mode]) == 3 for mode in MODES),
        "all_raw_runs_passed": len(results) == 9
        and all(result.get("pass") is True for result in results),
        "exact_semantics_reconstructed": exact_semantics,
        "route_coverage_reconstructed": route_coverage,
    }
    execution_pass = all(execution_checks.values())
    thresholds = workload.get("thresholds", {})
    qualification_pass = execution_pass and all(
        float(threshold_values.get(key, float("-inf"))) >= float(value)
        for key, value in thresholds.items()
    )
    checks = {
        **execution_checks,
        "input_hashes_match": input_hashes_match,
        "reported_values_match": values_match,
        "reported_execution_pass_matches": comparison.get("execution_pass")
        is execution_pass,
        "reported_qualification_pass_matches": comparison.get(
            "qualification_pass"
        )
        is qualification_pass,
        "formal_milestones_remain_open": comparison.get(
            "formal_milestones_closed"
        )
        == [],
    }
    return {
        "schema_version": 1,
        "gate": "independent M4a evidence verifier",
        "comparison": str(comparison_path),
        "reconstructed_primary_metrics": reconstructed,
        "reconstructed_threshold_values": threshold_values,
        "execution_pass": execution_pass,
        "qualification_pass": qualification_pass,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.result) != 9:
        parser.error("the verifier requires nine ordered --result paths")
    result = verify(
        args.comparison.resolve(strict=True),
        args.workload.resolve(strict=True),
        [path.resolve(strict=True) for path in args.result],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
