#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from vllm_ascend_resident_epoch.abi import (
    ABI_BYTES,
    NEW_TOTAL_BYTES,
    OLD_TOTAL_BYTES,
    TOTAL_REDUCTION_BYTES,
)


EXPECTED_ORDER = ("old", "new", "new", "old")
MEASURED_FIELDS = (
    "host_control_wall_us",
    "python_cpu_us",
    "native_wall_us",
    "native_cpu_us",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "q1": _percentile(values, 0.25),
        "median": statistics.median(values),
        "q3": _percentile(values, 0.75),
        "max": max(values),
    }


def _validate_block(block: dict[str, Any], expected_route: str) -> list[str]:
    failures: list[str] = []
    if block.get("pass") is not True:
        failures.append("block-not-passed")
    if block.get("route") != expected_route:
        failures.append("route-mismatch")
    samples = block.get("samples")
    if not isinstance(samples, list) or len(samples) != 15:
        failures.append("sample-count")
        return failures
    expected_bytes = ABI_BYTES[expected_route]
    for index, sample in enumerate(samples):
        prefix = f"sample-{index}"
        if sample.get("pass") is not True:
            failures.append(f"{prefix}-not-passed")
        expected = {
            "engine_core_step_calls": 1,
            "post_step_calls": 1,
            "socket_send_calls": 1,
            "socket_receive_calls": 1,
            "feed_calls": 1,
            "fetch_calls": 1,
            "model_calls": 2,
            "declared_input_bytes": expected_bytes["input"],
            "declared_output_bytes": expected_bytes["output"],
            "declared_total_bytes": expected_bytes["input"]
            + expected_bytes["output"],
        }
        for field, value in expected.items():
            if sample.get(field) != value:
                failures.append(f"{prefix}-{field}")
        for field in MEASURED_FIELDS:
            if not isinstance(sample.get(field), int) or sample[field] <= 0:
                failures.append(f"{prefix}-{field}")
    return failures


def analyze(
    block_paths: list[Path],
    semantic_path: Path,
    source_path: Path,
    profiler_path: Path,
    runtime_transfer_path: Path,
) -> dict[str, Any]:
    blocks = [_load(path) for path in block_paths]
    semantic = _load(semantic_path)
    source = _load(source_path)
    profiler = _load(profiler_path)
    runtime_transfer = _load(runtime_transfer_path)
    failures: list[str] = []
    for index, (block, route) in enumerate(zip(blocks, EXPECTED_ORDER, strict=True)):
        failures.extend(
            f"block-{index + 1}:{failure}"
            for failure in _validate_block(block, route)
        )
    if semantic.get("pass") is not True:
        failures.append("minimal-abi-multi-epoch-semantics")
    if source.get("pass") is not True:
        failures.append("minimal-abi-source-verification")
    if profiler.get("status") not in ("observed", "not_observed"):
        failures.append("invalid-profiler-status")
    if profiler.get("status") == "observed":
        for route in ("old", "new"):
            route_result = profiler.get("routes", {}).get(route, {})
            if not isinstance(route_result.get("total_bytes"), int):
                failures.append(f"profiler-{route}-bytes")
    elif not profiler.get("reason"):
        failures.append("profiler-not-observed-without-reason")
    if runtime_transfer.get("status") != "observed":
        failures.append("runtime-memcpy-not-observed")
    if runtime_transfer.get("logical_abi_bytes_used_as_transfer") is not False:
        failures.append("runtime-memcpy-used-logical-bytes")
    for route in ("old", "new"):
        route_result = runtime_transfer.get("routes", {}).get(route, {})
        if route_result.get("sample_count") != 30:
            failures.append(f"runtime-memcpy-{route}-sample-count")
        for field in (
            "host_to_device_bytes_median",
            "device_to_host_bytes_median",
            "total_bytes_median",
        ):
            value = route_result.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                failures.append(f"runtime-memcpy-{route}-{field}")

    samples = {
        "old": blocks[0].get("samples", []) + blocks[3].get("samples", []),
        "new": blocks[1].get("samples", []) + blocks[2].get("samples", []),
    }
    metrics: dict[str, Any] = {}
    if all(len(samples[route]) == 30 for route in samples):
        for field in MEASURED_FIELDS:
            old_values = [int(sample[field]) for sample in samples["old"]]
            new_values = [int(sample[field]) for sample in samples["new"]]
            old_stats = _stats(old_values)
            new_stats = _stats(new_values)
            metrics[field] = {
                "old": old_stats,
                "new": new_stats,
                "old_over_new_median": (
                    old_stats["median"] / new_stats["median"]
                    if new_stats["median"]
                    else None
                ),
                "new_minus_old_median": new_stats["median"]
                - old_stats["median"],
            }

    control_variables = {
        "baseline_sha256": sorted(
            {block.get("artifacts", {}).get("baseline_sha256") for block in blocks}
        ),
        "resolved_classes": [block.get("resolved_classes") for block in blocks],
        "repetitions": [block.get("repetitions") for block in blocks],
    }
    if len(control_variables["baseline_sha256"]) != 1:
        failures.append("baseline-hash-changed")
    if len({json.dumps(value, sort_keys=True) for value in control_variables["resolved_classes"]}) != 1:
        failures.append("resolved-classes-changed")
    if control_variables["repetitions"] != [15, 15, 15, 15]:
        failures.append("repetition-budget-changed")

    return {
        "gate": "Attempt 74 minimal Host-UDF ABI and control-overhead evidence",
        "schema_version": 1,
        "pass": not failures,
        "failed_checks": sorted(set(failures)),
        "block_order": list(EXPECTED_ORDER),
        "control_variables": control_variables,
        "logical_abi": {
            "old_input_bytes": ABI_BYTES["old"]["input"],
            "old_output_bytes": ABI_BYTES["old"]["output"],
            "old_total_bytes": OLD_TOTAL_BYTES,
            "new_input_bytes": ABI_BYTES["new"]["input"],
            "new_output_bytes": ABI_BYTES["new"]["output"],
            "new_total_bytes": NEW_TOTAL_BYTES,
            "reduction_bytes": TOTAL_REDUCTION_BYTES,
            "reduction_fraction": TOTAL_REDUCTION_BYTES / OLD_TOTAL_BYTES,
        },
        "host_api_per_epoch": {
            "engine_core_step": 1,
            "post_step": 1,
            "socket_send": 1,
            "socket_receive": 1,
            "dataflow_feed": 1,
            "dataflow_fetch": 1,
            "device_model_calls": 2,
        },
        "metrics": metrics,
        "profiler_transfer": profiler,
        "runtime_memcpy_transfer": runtime_transfer,
        "semantic_result": {
            "path": str(semantic_path),
            "sha256": _sha256(semantic_path),
            "pass": semantic.get("pass"),
        },
        "source_verification": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "pass": source.get("pass"),
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256(path), "route": route}
            for path, route in zip(block_paths, EXPECTED_ORDER, strict=True)
        ],
        "profiler_summary": {
            "path": str(profiler_path),
            "sha256": _sha256(profiler_path),
        },
        "runtime_memcpy_summary": {
            "path": str(runtime_transfer_path),
            "sha256": _sha256(runtime_transfer_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-1", type=Path, required=True)
    parser.add_argument("--new-1", type=Path, required=True)
    parser.add_argument("--new-2", type=Path, required=True)
    parser.add_argument("--old-2", type=Path, required=True)
    parser.add_argument("--semantic-result", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument("--profiler-summary", type=Path, required=True)
    parser.add_argument("--runtime-memcpy-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.old_1, args.new_1, args.new_2, args.old_2]
    try:
        result = analyze(
            paths,
            args.semantic_result,
            args.source_verification,
            args.profiler_summary,
            args.runtime_memcpy_summary,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ABI_COMPARISON_INVALID: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
