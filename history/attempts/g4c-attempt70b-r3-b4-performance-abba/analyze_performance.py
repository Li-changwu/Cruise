#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


KS = (2, 4, 8)
WARMUPS = 3
REPEATS = 15


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "q1": percentile(values, 0.25),
        "median": statistics.median(values),
        "q3": percentile(values, 0.75),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def validate_pairs(
    rows: list[dict[str, str | int]], count: int, k: int
) -> tuple[list[tuple[dict[str, str | int], dict[str, str | int]]], bool, bool]:
    pairs = []
    order_pass = True
    submission_pass = True
    for iteration in range(count):
        pair = sorted(
            [row for row in rows if row["iteration"] == iteration],
            key=lambda row: row["position"],
        )
        expected_order = "HD" if iteration % 2 == 0 else "DH"
        expected_routes = (
            ["host", "device"] if expected_order == "HD" else ["device", "host"]
        )
        if (
            len(pair) != 2
            or any(row["order"] != expected_order for row in pair)
            or [row["route"] for row in pair] != expected_routes
        ):
            order_pass = False
            continue
        by_route = {row["route"]: row for row in pair}
        host = by_route["host"]
        device = by_route["device"]
        submission_pass &= (
            host["host_model_submissions"] == k
            and host["feed_calls"] == 0
            and host["fetch_calls"] == 0
            and device["host_model_submissions"] == 1
            and device["feed_calls"] == 1
            and device["fetch_calls"] == 1
        )
        pairs.append((host, device))
    return pairs, order_pass, submission_pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    numeric = (
        "k",
        "iteration",
        "position",
        "wall_us",
        "cpu_us",
        "host_model_submissions",
        "feed_calls",
        "fetch_calls",
    )
    for row in rows:
        for key in numeric:
            row[key] = int(row[key])

    cases = []
    for k in KS:
        warmups = [row for row in rows if row["phase"] == "warmup" and row["k"] == k]
        measured = [row for row in rows if row["phase"] == "measure" and row["k"] == k]
        warmup_pairs, warmup_order_pass, warmup_submission_pass = validate_pairs(
            warmups, WARMUPS, k
        )
        pairs, order_pass, submission_pass = validate_pairs(measured, REPEATS, k)

        host_wall = [host["wall_us"] for host, _ in pairs]
        device_wall = [device["wall_us"] for _, device in pairs]
        host_cpu = [host["cpu_us"] for host, _ in pairs]
        device_cpu = [device["cpu_us"] for _, device in pairs]
        speedups = [host["wall_us"] / device["wall_us"] for host, device in pairs]
        host_summary = summarize(host_wall) if host_wall else {}
        device_summary = summarize(device_wall) if device_wall else {}
        host_cpu_summary = summarize(host_cpu) if host_cpu else {}
        device_cpu_summary = summarize(device_cpu) if device_cpu else {}
        median_speedup = statistics.median(speedups) if speedups else 0.0
        faster_count = sum(device < host for host, device in zip(host_wall, device_wall))
        separated_iqr = bool(
            host_wall
            and device_wall
            and device_summary["q3"] < host_summary["q1"]
        )
        case_pass = bool(
            len(warmup_pairs) == WARMUPS
            and warmup_order_pass
            and warmup_submission_pass
            and len(pairs) == REPEATS
            and order_pass
            and submission_pass
            and all(value > 0 for value in host_wall + device_wall + host_cpu + device_cpu)
            and device_summary["median"] < host_summary["median"]
            and device_cpu_summary["median"] < host_cpu_summary["median"]
            and median_speedup >= 1.10
            and faster_count >= 13
            and separated_iqr
        )
        cases.append(
            {
                "k": k,
                "warmup_rows": len(warmups),
                "warmup_pairs": len(warmup_pairs),
                "warmup_alternating_order": warmup_order_pass,
                "warmup_submission_semantics": warmup_submission_pass,
                "measured_pairs": len(pairs),
                "alternating_order": order_pass,
                "submission_semantics": submission_pass,
                "host_wall_us": host_summary,
                "device_wall_us": device_summary,
                "host_cpu_us": host_cpu_summary,
                "device_cpu_us": device_cpu_summary,
                "paired_speedup_median": median_speedup,
                "device_faster_pairs": faster_count,
                "iqr_separated": separated_iqr,
                "pass": case_pass,
            }
        )

    expected_rows = len(KS) * (WARMUPS + REPEATS) * 2
    result = {
        "gate": "G4c Attempt 70b B=4 stable performance",
        "pass": len(rows) == expected_rows and all(case["pass"] for case in cases),
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "thresholds": {
            "warmups_per_route_per_k": WARMUPS,
            "measured_pairs_per_k": REPEATS,
            "minimum_median_speedup": 1.10,
            "minimum_device_faster_pairs": 13,
            "require_non_overlapping_iqr": True,
            "require_device_cpu_median_lower": True,
        },
        "cases": cases,
        "claim_boundary": (
            "This measures fixed B=4 K=2/4/8 epochs with both routes resident "
            "in one process. It does not include scheduler integration or dynamic batching."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
