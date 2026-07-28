#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any


DIRECTIONS = {
    1: "host_to_device",
    2: "device_to_host",
    6: "host_to_device",
    7: "device_to_host",
}
BLOCKS = ("old-1", "new-1", "new-2", "old-2")
ROUTES = {"old-1": "old", "new-1": "new", "new-2": "new", "old-2": "old"}
TRACE_FIELDS = (
    "api",
    "pid",
    "tid",
    "start_ns",
    "end_ns",
    "bytes",
    "dest_max",
    "kind",
    "status",
)
TRACE_APIS = ("rtMemcpy", "rtMemcpyAsync")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_trace(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TRACE_FIELDS:
            raise ValueError(f"unexpected rtMemcpy trace header in {path}")
        for line, row in enumerate(reader, start=2):
            try:
                parsed = {field: int(row[field]) for field in TRACE_FIELDS[1:]}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid rtMemcpy trace row {path}:{line}") from exc
            api = row.get("api")
            if api not in TRACE_APIS:
                raise ValueError(f"invalid rtMemcpy API {path}:{line}")
            parsed["api"] = api
            if parsed["start_ns"] <= 0 or parsed["end_ns"] < parsed["start_ns"]:
                raise ValueError(f"invalid rtMemcpy timestamp {path}:{line}")
            if parsed["bytes"] < 0 or parsed["dest_max"] < parsed["bytes"]:
                raise ValueError(f"invalid rtMemcpy byte count {path}:{line}")
            rows.append(parsed)
    return rows


def _load_windows(path: Path) -> list[tuple[int, int]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    samples = result.get("samples")
    if not isinstance(samples, list) or len(samples) != 15:
        raise ValueError(f"benchmark result must contain 15 samples: {path}")
    windows: list[tuple[int, int]] = []
    for index, sample in enumerate(samples):
        start = sample.get("epoch_wall_clock_start_ns")
        end = sample.get("epoch_wall_clock_end_ns")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise ValueError(f"invalid epoch wall-clock window {path}:{index}")
        if windows and start <= windows[-1][1]:
            raise ValueError(f"overlapping epoch wall-clock windows: {path}")
        windows.append((start, end))
    return windows


def _summarize_block(
    label: str,
    trace_path: Path,
    result_path: Path,
    filtered_path: Path,
) -> dict[str, Any]:
    calls = _read_trace(trace_path)
    windows = _load_windows(result_path)
    samples: list[dict[str, Any]] = []
    boundary_crossings: list[dict[str, int | str]] = []
    filtered_rows: list[dict[str, Any]] = []
    failed_directional_calls = 0
    for sample_index, (window_start, window_end) in enumerate(windows):
        totals = {"host_to_device": 0, "device_to_host": 0}
        counts = {"host_to_device": 0, "device_to_host": 0}
        for call in calls:
            overlaps = call["end_ns"] >= window_start and call["start_ns"] <= window_end
            contained = call["start_ns"] >= window_start and call["end_ns"] <= window_end
            if overlaps and not contained:
                boundary_crossings.append(call)
                continue
            if not contained:
                continue
            direction = DIRECTIONS.get(call["kind"])
            if direction is None:
                continue
            filtered_rows.append(
                {
                    "block": label,
                    "sample": sample_index,
                    "direction": direction,
                    **call,
                }
            )
            if call["status"] != 0:
                failed_directional_calls += 1
                continue
            totals[direction] += call["bytes"]
            counts[direction] += 1
        observed = all(counts[direction] > 0 for direction in totals)
        samples.append(
            {
                "sample": sample_index,
                "window_start_ns": window_start,
                "window_end_ns": window_end,
                "host_to_device_calls": counts["host_to_device"],
                "device_to_host_calls": counts["device_to_host"],
                "host_to_device_bytes": totals["host_to_device"],
                "device_to_host_bytes": totals["device_to_host"],
                "total_bytes": sum(totals.values()),
                "observed_both_directions": observed,
            }
        )

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("block", "sample", "direction", *TRACE_FIELDS)
    with filtered_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(filtered_rows)

    observed = (
        all(sample["observed_both_directions"] for sample in samples)
        and not boundary_crossings
        and failed_directional_calls == 0
    )
    return {
        "route": ROUTES[label],
        "status": "observed" if observed else "not_observed",
        "reason": None
        if observed
        else (
            "every measured epoch must contain successful H2D and D2H rtMemcpy "
            "calls with no call crossing an epoch boundary"
        ),
        "trace_path": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "benchmark_result_path": str(result_path),
        "benchmark_result_sha256": _sha256(result_path),
        "filtered_trace_path": str(filtered_path),
        "filtered_trace_sha256": _sha256(filtered_path),
        "raw_call_count": len(calls),
        "filtered_directional_call_count": len(filtered_rows),
        "failed_directional_calls": failed_directional_calls,
        "boundary_crossing_calls": len(boundary_crossings),
        "samples": samples,
    }


def summarize(
    traces: dict[str, Path],
    results: dict[str, Path],
    filtered_dir: Path,
) -> dict[str, Any]:
    blocks = {
        label: _summarize_block(
            label,
            traces[label],
            results[label],
            filtered_dir / f"{label}.tsv",
        )
        for label in BLOCKS
    }
    status = (
        "observed"
        if all(block["status"] == "observed" for block in blocks.values())
        else "not_observed"
    )
    route_samples = {
        "old": blocks["old-1"]["samples"] + blocks["old-2"]["samples"],
        "new": blocks["new-1"]["samples"] + blocks["new-2"]["samples"],
    }
    route_summary: dict[str, Any] = {}
    for route, samples in route_samples.items():
        route_summary[route] = {
            "sample_count": len(samples),
            "host_to_device_bytes_median": statistics.median(
                sample["host_to_device_bytes"] for sample in samples
            ),
            "device_to_host_bytes_median": statistics.median(
                sample["device_to_host_bytes"] for sample in samples
            ),
            "total_bytes_median": statistics.median(
                sample["total_bytes"] for sample in samples
            ),
        }
    return {
        "schema_version": 1,
        "metric": "successful CANN runtime memcpy bytes inside measured EngineCore epoch windows",
        "measurement_boundary": (
            "LD_PRELOAD interception of libge_executor rtMemcpy/rtMemcpyAsync calls in the "
            "resident sidecar; counts runtime copy requests, not declared ABI bytes"
        ),
        "status": status,
        "reason": None
        if status == "observed"
        else "; ".join(
            f"{label}: {block['reason']}"
            for label, block in blocks.items()
            if block["status"] != "observed"
        ),
        "logical_abi_bytes_used_as_transfer": False,
        "clock": "CLOCK_REALTIME/time.time_ns",
        "blocks": blocks,
        "routes": route_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for label in BLOCKS:
        option = label.replace("-", "_")
        parser.add_argument(f"--{label}-trace", dest=f"{option}_trace", type=Path, required=True)
        parser.add_argument(f"--{label}-result", dest=f"{option}_result", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = {
        label: getattr(args, f"{label.replace('-', '_')}_trace") for label in BLOCKS
    }
    results = {
        label: getattr(args, f"{label.replace('-', '_')}_result") for label in BLOCKS
    }
    try:
        result = summarize(traces, results, args.filtered_dir)
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        print(f"RUNTIME_MEMCPY_SUMMARY_INVALID: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
