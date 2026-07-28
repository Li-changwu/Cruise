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


EXPECTED_ORDER = ["old", "new", "new", "old"]
EXPECTED_LOGICAL = {
    "old_input_bytes": 58_720_516,
    "old_output_bytes": 78_184_928,
    "old_total_bytes": 136_905_444,
    "new_input_bytes": 260,
    "new_output_bytes": 368,
    "new_total_bytes": 628,
    "reduction_bytes": 136_904_816,
}
EXPECTED_APIS = {
    "engine_core_step": 1,
    "post_step": 1,
    "socket_send": 1,
    "socket_receive": 1,
    "dataflow_feed": 1,
    "dataflow_fetch": 1,
    "device_model_calls": 2,
}
FIELDS = (
    "host_control_wall_us",
    "python_cpu_us",
    "native_wall_us",
    "native_cpu_us",
)
RUNTIME_TRACE_FIELDS = (
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
FILTERED_RUNTIME_TRACE_FIELDS = ("block", "sample", "direction", *RUNTIME_TRACE_FIELDS)
RUNTIME_APIS = {"rtMemcpy", "rtMemcpyAsync"}
RUNTIME_DIRECTIONS = {
    1: "host_to_device",
    2: "device_to_host",
    6: "host_to_device",
    7: "device_to_host",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hashed(entry: dict[str, Any], label: str) -> dict[str, Any]:
    path = Path(entry.get("path", ""))
    _require(path.is_file(), f"{label}: missing input file")
    _require(_sha256(path) == entry.get("sha256"), f"{label}: SHA256 mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_result(data: dict[str, Any], *, require_artifacts: bool) -> dict[str, Any]:
    _require(
        data.get("gate")
        == "Attempt 74 minimal Host-UDF ABI and control-overhead evidence",
        "unexpected gate",
    )
    _require(data.get("schema_version") == 1, "unexpected schema version")
    _require(data.get("pass") is True, "analysis did not pass")
    _require(data.get("failed_checks") == [], "analysis retained failed checks")
    _require(data.get("block_order") == EXPECTED_ORDER, "block order is not ABBA")
    logical = data.get("logical_abi")
    _require(isinstance(logical, dict), "missing logical ABI ledger")
    for key, expected in EXPECTED_LOGICAL.items():
        _require(logical.get(key) == expected, f"logical ABI changed: {key}")
    _require(logical.get("reduction_fraction", 0) > 0.999, "ABI reduction too small")
    _require(data.get("host_api_per_epoch") == EXPECTED_APIS, "Host API ledger changed")

    inputs = data.get("inputs")
    _require(isinstance(inputs, list) and len(inputs) == 4, "expected four raw blocks")
    raw_blocks: list[dict[str, Any]] = []
    for index, (entry, route) in enumerate(zip(inputs, EXPECTED_ORDER, strict=True)):
        _require(entry.get("route") == route, f"block {index}: route changed")
        block = _load_hashed(entry, f"block {index}") if require_artifacts else {}
        if require_artifacts:
            _require(block.get("pass") is True, f"block {index}: raw run failed")
            _require(block.get("route") == route, f"block {index}: raw route changed")
            _require(len(block.get("samples", [])) == 15, f"block {index}: sample count")
            raw_blocks.append(block)

    metrics = data.get("metrics")
    _require(isinstance(metrics, dict), "missing metric breakdown")
    for field in FIELDS:
        metric = metrics.get(field)
        _require(isinstance(metric, dict), f"missing metric {field}")
        for route in ("old", "new"):
            stats = metric.get(route)
            _require(isinstance(stats, dict), f"missing {route} {field} stats")
            _require(stats.get("count") == 30, f"{route} {field}: wrong count")
            _require(stats.get("median", 0) > 0, f"{route} {field}: invalid median")
        _require(
            isinstance(metric.get("old_over_new_median"), (int, float)),
            f"missing {field} ratio",
        )

    if require_artifacts:
        route_samples = {
            "old": raw_blocks[0]["samples"] + raw_blocks[3]["samples"],
            "new": raw_blocks[1]["samples"] + raw_blocks[2]["samples"],
        }
        for field in FIELDS:
            for route in ("old", "new"):
                median = statistics.median(
                    int(sample[field]) for sample in route_samples[route]
                )
                _require(
                    metrics[field][route]["median"] == median,
                    f"{route} {field}: median does not match raw rows",
                )

    for key in ("semantic_result", "source_verification"):
        entry = data.get(key)
        _require(isinstance(entry, dict) and entry.get("pass") is True, f"{key} failed")
        if require_artifacts:
            artifact = _load_hashed(entry, key)
            _require(artifact.get("pass") is True, f"{key} artifact failed")

    profiler = data.get("profiler_transfer")
    _require(isinstance(profiler, dict), "missing profiler transfer result")
    _require(
        profiler.get("logical_abi_bytes_used_as_transfer") is False,
        "logical ABI bytes were substituted for profiler bytes",
    )
    status = profiler.get("status")
    _require(status in ("observed", "not_observed"), "invalid profiler status")
    if status == "observed":
        for route in ("old", "new"):
            value = profiler.get("routes", {}).get(route, {}).get("total_bytes")
            _require(isinstance(value, int) and value >= 0, f"{route}: invalid profiler bytes")
    else:
        _require(bool(profiler.get("reason")), "not_observed has no reason")

    runtime_transfer = data.get("runtime_memcpy_transfer")
    _require(isinstance(runtime_transfer, dict), "missing runtime memcpy transfer")
    _require(
        runtime_transfer.get("logical_abi_bytes_used_as_transfer") is False,
        "logical ABI bytes were substituted for runtime memcpy bytes",
    )
    _require(
        runtime_transfer.get("status") == "observed",
        "runtime memcpy bytes were not observed for every measured epoch",
    )
    blocks = runtime_transfer.get("blocks")
    _require(isinstance(blocks, dict), "missing runtime memcpy blocks")
    for label, route in zip(("old-1", "new-1", "new-2", "old-2"), EXPECTED_ORDER):
        block = blocks.get(label)
        _require(isinstance(block, dict), f"missing runtime memcpy block {label}")
        _require(block.get("route") == route, f"runtime memcpy route changed: {label}")
        _require(block.get("status") == "observed", f"runtime memcpy block failed: {label}")
        _require(len(block.get("samples", [])) == 15, f"runtime memcpy sample count: {label}")
        for sample in block["samples"]:
            _require(
                sample.get("observed_both_directions") is True,
                f"runtime memcpy direction missing: {label}",
            )
            _require(sample.get("host_to_device_bytes", 0) > 0, f"missing H2D bytes: {label}")
            _require(sample.get("device_to_host_bytes", 0) > 0, f"missing D2H bytes: {label}")
        if require_artifacts:
            filtered = Path(block.get("filtered_trace_path", ""))
            _require(filtered.is_file(), f"missing filtered runtime trace: {label}")
            _require(
                _sha256(filtered) == block.get("filtered_trace_sha256"),
                f"filtered runtime trace SHA256 mismatch: {label}",
            )
            recomputed = {
                index: {
                    "host_to_device_calls": 0,
                    "device_to_host_calls": 0,
                    "host_to_device_bytes": 0,
                    "device_to_host_bytes": 0,
                }
                for index in range(15)
            }
            with filtered.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                _require(
                    tuple(reader.fieldnames or ()) == FILTERED_RUNTIME_TRACE_FIELDS,
                    f"unexpected filtered runtime trace header: {label}",
                )
                for row in reader:
                    sample_index = int(row["sample"])
                    _require(sample_index in recomputed, f"invalid filtered sample: {label}")
                    _require(row["block"] == label, f"filtered block mismatch: {label}")
                    _require(row["api"] in RUNTIME_APIS, f"invalid filtered API: {label}")
                    direction = row["direction"]
                    _require(
                        direction in ("host_to_device", "device_to_host"),
                        f"invalid filtered direction: {label}",
                    )
                    start_ns = int(row["start_ns"])
                    end_ns = int(row["end_ns"])
                    byte_count = int(row["bytes"])
                    dest_max = int(row["dest_max"])
                    kind = int(row["kind"])
                    _require(
                        start_ns > 0 and end_ns >= start_ns,
                        f"invalid filtered timestamps: {label}",
                    )
                    _require(
                        byte_count >= 0 and dest_max >= byte_count,
                        f"invalid filtered byte count: {label}",
                    )
                    _require(
                        RUNTIME_DIRECTIONS.get(kind) == direction,
                        f"filtered direction/kind mismatch: {label}",
                    )
                    _require(int(row["status"]) == 0, f"failed filtered copy: {label}")
                    recomputed[sample_index][f"{direction}_calls"] += 1
                    recomputed[sample_index][f"{direction}_bytes"] += byte_count
            for sample in block["samples"]:
                values = recomputed[sample["sample"]]
                for field, value in values.items():
                    _require(sample.get(field) == value, f"filtered {field} mismatch: {label}")
                _require(
                    sample.get("total_bytes")
                    == values["host_to_device_bytes"] + values["device_to_host_bytes"],
                    f"filtered total bytes mismatch: {label}",
                )

    runtime_entry = data.get("runtime_memcpy_summary")
    _require(isinstance(runtime_entry, dict), "missing runtime memcpy summary input")
    if require_artifacts:
        runtime_artifact = _load_hashed(runtime_entry, "runtime_memcpy_summary")
        _require(runtime_artifact.get("status") == "observed", "runtime memcpy artifact failed")

    return {
        "pass": True,
        "raw_blocks": 4,
        "samples_per_route": 30,
        "old_declared_bytes_per_epoch": 136_905_444,
        "new_declared_bytes_per_epoch": 628,
        "profiler_transfer_status": status,
        "runtime_memcpy_transfer_status": runtime_transfer["status"],
        "host_control_metrics": list(FIELDS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--no-require-artifacts", action="store_true")
    args = parser.parse_args()
    try:
        data = json.loads(args.result.read_text(encoding="utf-8"))
        summary = validate_result(
            data, require_artifacts=not args.no_require_artifacts
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ABI_COMPARISON_RESULT_INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
