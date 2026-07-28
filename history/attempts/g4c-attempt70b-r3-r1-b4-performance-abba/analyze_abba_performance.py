#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


KS = (2, 4, 8)
WARMUPS_PER_BLOCK = 3
BLOCKS = {
    "h1": {"route": "host", "block": 1, "repeats": 8, "order": "HD", "position": 0},
    "d1": {"route": "device", "block": 1, "repeats": 8, "order": "HD", "position": 1},
    "d2": {"route": "device", "block": 2, "repeats": 7, "order": "DH", "position": 0},
    "h2": {"route": "host", "block": 2, "repeats": 7, "order": "DH", "position": 1},
}
NUMERIC = (
    "block",
    "k",
    "iteration",
    "position",
    "wall_us",
    "cpu_us",
    "host_model_submissions",
    "feed_calls",
    "fetch_calls",
)


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


def submission_ok(row: dict[str, str | int], k: int) -> bool:
    if row["route"] == "host":
        return (
            row["host_model_submissions"] == k
            and row["feed_calls"] == 0
            and row["fetch_calls"] == 0
        )
    return (
        row["host_model_submissions"] == 1
        and row["feed_calls"] == 1
        and row["fetch_calls"] == 1
    )


def read_block(path: Path, name: str) -> tuple[list[dict[str, str | int]], bool]:
    spec = BLOCKS[name]
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for key in NUMERIC:
            row[key] = int(row[key])
    expected_rows = len(KS) * (WARMUPS_PER_BLOCK + spec["repeats"])
    structural = len(rows) == expected_rows and all(
        row["block"] == spec["block"]
        and row["route"] == spec["route"]
        and row["order"] == spec["order"]
        and row["position"] == spec["position"]
        for row in rows
    )
    return rows, structural


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in BLOCKS:
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = {name: getattr(args, name) for name in BLOCKS}
    rows_by_block = {}
    structural_by_block = {}
    for name, path in paths.items():
        rows_by_block[name], structural_by_block[name] = read_block(path, name)
    chronological_files = all(
        paths[left].stat().st_mtime_ns <= paths[right].stat().st_mtime_ns
        for left, right in zip(BLOCKS, tuple(BLOCKS)[1:])
    )
    rows = [row for name in BLOCKS for row in rows_by_block[name]]

    cases = []
    for k in KS:
        warmup_pass = True
        for name, spec in BLOCKS.items():
            selected = [
                row
                for row in rows_by_block[name]
                if row["phase"] == "warmup" and row["k"] == k
            ]
            warmup_pass &= (
                len(selected) == WARMUPS_PER_BLOCK
                and sorted(row["iteration"] for row in selected)
                == list(range(WARMUPS_PER_BLOCK))
                and all(submission_ok(row, k) for row in selected)
                and all(row["wall_us"] > 0 and row["cpu_us"] > 0 for row in selected)
            )

        pairs = []
        order_pass = True
        submission_pass = True
        for iteration in range(15):
            block = 1 if iteration < 8 else 2
            expected_order = "HD" if block == 1 else "DH"
            pair = [
                row
                for row in rows
                if row["phase"] == "measure"
                and row["k"] == k
                and row["iteration"] == iteration
            ]
            by_route = {row["route"]: row for row in pair}
            if len(pair) != 2 or set(by_route) != {"host", "device"}:
                order_pass = False
                continue
            host = by_route["host"]
            device = by_route["device"]
            order_pass &= (
                host["block"] == block
                and device["block"] == block
                and host["order"] == expected_order
                and device["order"] == expected_order
                and host["position"] == (0 if block == 1 else 1)
                and device["position"] == (1 if block == 1 else 0)
            )
            submission_pass &= submission_ok(host, k) and submission_ok(device, k)
            pairs.append((host, device))

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
            host_wall and device_wall and device_summary["q3"] < host_summary["q1"]
        )
        case_pass = bool(
            warmup_pass
            and len(pairs) == 15
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
                "warmup_blocks_valid": warmup_pass,
                "measured_pairs": len(pairs),
                "abba_order": order_pass,
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

    expected_rows = len(KS) * 42
    result = {
        "gate": "G4c Attempt 70b-r3-r1 B=4 stable ABBA performance",
        "pass": (
            len(rows) == expected_rows
            and chronological_files
            and all(structural_by_block.values())
            and all(case["pass"] for case in cases)
        ),
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "block_file_order": list(BLOCKS),
        "block_files_chronological": chronological_files,
        "block_structure": structural_by_block,
        "thresholds": {
            "warmups_per_route_per_block_per_k": WARMUPS_PER_BLOCK,
            "measured_pairs_per_k": 15,
            "minimum_median_speedup": 1.10,
            "minimum_device_faster_pairs": 13,
            "require_non_overlapping_iqr": True,
            "require_device_cpu_median_lower": True,
        },
        "cases": cases,
        "claim_boundary": (
            "This measures fixed B=4 K=2/4/8 epochs in H1-D1-D2-H2 ABBA "
            "single-route resident sessions. Initialization and warmup are excluded. "
            "It does not include scheduler integration or dynamic batching."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
