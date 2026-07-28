#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_ORDER = ["old", "new", "new", "old"]
BLOCK_LABELS = ("old-1", "new-1", "new-2", "old-2")
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
EXPECTED_TENSOR_BYTES = {
    "old": {
        "host_to_device": [
            32,
            32,
            16,
            29_360_128,
            16,
            16,
            32,
            29_360_128,
            72,
            44,
        ],
        "device_to_host": [
            19_464_192,
            256,
            29_360_128,
            29_360_128,
            32,
            32,
            16,
            16,
            16,
            112,
        ],
    },
    "new": {
        "host_to_device": [32, 32, 16, 16, 16, 32, 72, 44],
        "device_to_host": [256, 112],
    },
}
FIELDS = (
    "host_control_wall_us",
    "python_cpu_us",
    "native_wall_us",
    "native_cpu_us",
)
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
FILTERED_TRACE_FIELDS = (
    "block",
    "sample",
    "transport",
    "direction",
    *TRACE_FIELDS,
)
RUNTIME_MEMCPY_APIS = frozenset(
    {
        "rtMemcpy",
        "rtMemcpyEx",
        "rtMemcpyAsync",
        "rtMemcpyAsyncWithoutCheckKind",
        "rtMemcpyAsyncEx",
        "rtMemcpyAsyncWithCfg",
        "rtMemcpyAsyncWithCfgV2",
        "rtMemcpyHostTask",
        "rtMemcpyAsyncWithOffset",
        "rtMemcpy2d",
        "rtMemcpy2dAsync",
        "rtsMemcpy",
        "rtsMemcpyAsync",
        "rtsMemcpyBatch",
        "rtsMemcpyBatchAsync",
        "rtsMemcpyAsyncWithDesc",
    }
)
MBUF_DIAGNOSTIC_APIS = frozenset(
    {
        "rtMbufAlloc",
        "rtMbufBuild",
        "rtMbufSetDataLen",
        "rtMbufGetBuffSize",
        "rtBuffGet",
        "rtBuffConfirm",
    }
)
DATAFLOW_API_DIRECTIONS = {
    "FeedDataFlowGraphTensor": "host_to_device",
    "FetchDataFlowGraphTensor": "device_to_host",
}
TRACE_APIS = RUNTIME_MEMCPY_APIS | MBUF_DIAGNOSTIC_APIS | frozenset(
    DATAFLOW_API_DIRECTIONS
)


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


def _transport(api: str) -> str:
    if api in DATAFLOW_API_DIRECTIONS:
        return "dataflow_tensor"
    if api in RUNTIME_MEMCPY_APIS:
        return "runtime_memcpy"
    if api in MBUF_DIAGNOSTIC_APIS:
        return "mbuf_diagnostic"
    raise ValueError(f"unclassified trace API: {api}")


def _runtime_direction(api: str, kind: int) -> str | None:
    if kind == 1:
        return "host_to_device"
    if kind == 2:
        return "device_to_host"
    if not api.startswith("rts") and kind == 6:
        return "host_to_device"
    if not api.startswith("rts") and kind == 7:
        return "device_to_host"
    return None


def _direction(api: str, kind: int) -> str:
    if api in DATAFLOW_API_DIRECTIONS:
        return DATAFLOW_API_DIRECTIONS[api]
    if api in RUNTIME_MEMCPY_APIS:
        return _runtime_direction(api, kind) or ""
    return ""


def _read_trace(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require(tuple(reader.fieldnames or ()) == TRACE_FIELDS, "unexpected raw trace header")
        for line, row in enumerate(reader, start=2):
            api = row.get("api", "")
            _require(api in TRACE_APIS, f"invalid raw trace API at line {line}")
            try:
                parsed: dict[str, int | str] = {
                    "api": api,
                    **{field: int(row[field]) for field in TRACE_FIELDS[1:]},
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid raw trace row at line {line}") from exc
            _require(
                int(parsed["start_ns"]) > 0
                and int(parsed["end_ns"]) >= int(parsed["start_ns"]),
                f"invalid raw trace timestamp at line {line}",
            )
            _require(
                int(parsed["bytes"]) >= 0
                and int(parsed["dest_max"]) >= int(parsed["bytes"]),
                f"invalid raw trace byte count at line {line}",
            )
            rows.append(parsed)
    return rows


def _call_count(rows: list[dict[str, int | str]], api: str) -> int:
    return len(
        {
            (
                row["pid"],
                row["tid"],
                row["start_ns"],
                row["end_ns"],
                row["status"],
            )
            for row in rows
            if row["api"] == api
        }
    )


def _normalize_filtered_row(row: dict[str, Any]) -> dict[str, str]:
    return {field: str(row.get(field, "")) for field in FILTERED_TRACE_FIELDS}


def _recompute_transfer_block(
    label: str,
    route: str,
    transfer_block: dict[str, Any],
    raw_block: dict[str, Any],
) -> list[dict[str, Any]]:
    trace_path = Path(transfer_block.get("trace_path", ""))
    _require(trace_path.is_file(), f"missing raw transfer trace: {label}")
    _require(
        _sha256(trace_path) == transfer_block.get("trace_sha256"),
        f"raw transfer trace SHA256 mismatch: {label}",
    )
    benchmark_path = Path(transfer_block.get("benchmark_result_path", ""))
    _require(benchmark_path.is_file(), f"missing benchmark result: {label}")
    _require(
        _sha256(benchmark_path) == transfer_block.get("benchmark_result_sha256"),
        f"benchmark result SHA256 mismatch: {label}",
    )
    _require(
        json.loads(benchmark_path.read_text(encoding="utf-8")) == raw_block,
        f"benchmark result does not match analyzer input: {label}",
    )

    calls = _read_trace(trace_path)
    raw_samples = raw_block["samples"]
    expected_filtered: list[dict[str, str]] = []
    measured_indexes: set[int] = set()
    boundary_record_indexes: set[int] = set()
    boundary_call_keys: set[tuple[Any, ...]] = set()
    recomputed_samples: list[dict[str, Any]] = []

    for sample_index, raw_sample in enumerate(raw_samples):
        window_start = int(raw_sample["epoch_wall_clock_start_ns"])
        window_end = int(raw_sample["epoch_wall_clock_end_ns"])
        _require(window_end > window_start, f"invalid raw epoch window: {label}")
        contained: list[dict[str, int | str]] = []
        for call_index, call in enumerate(calls):
            overlaps = int(call["end_ns"]) >= window_start and int(
                call["start_ns"]
            ) <= window_end
            is_contained = int(call["start_ns"]) >= window_start and int(
                call["end_ns"]
            ) <= window_end
            if overlaps and not is_contained:
                boundary_record_indexes.add(call_index)
                boundary_call_keys.add(
                    (
                        _transport(str(call["api"])),
                        call["api"],
                        call["pid"],
                        call["tid"],
                        call["start_ns"],
                        call["end_ns"],
                    )
                )
                continue
            if not is_contained:
                continue
            measured_indexes.add(call_index)
            contained.append(call)
            api = str(call["api"])
            expected_filtered.append(
                _normalize_filtered_row(
                    {
                        "block": label,
                        "sample": sample_index,
                        "transport": _transport(api),
                        "direction": _direction(api, int(call["kind"])),
                        **call,
                    }
                )
            )

        feed = [row for row in contained if row["api"] == "FeedDataFlowGraphTensor"]
        fetch = [row for row in contained if row["api"] == "FetchDataFlowGraphTensor"]
        runtime = [row for row in contained if row["api"] in RUNTIME_MEMCPY_APIS]
        mbuf = [row for row in contained if row["api"] in MBUF_DIAGNOSTIC_APIS]
        feed_sizes = [int(row["bytes"]) for row in feed]
        fetch_sizes = [int(row["bytes"]) for row in fetch]
        _require(
            feed_sizes == EXPECTED_TENSOR_BYTES[route]["host_to_device"],
            f"DataFlow Feed tensor payload mismatch: {label}/{sample_index}",
        )
        _require(
            fetch_sizes == EXPECTED_TENSOR_BYTES[route]["device_to_host"],
            f"DataFlow Fetch tensor payload mismatch: {label}/{sample_index}",
        )
        _require(
            _call_count(contained, "FeedDataFlowGraphTensor") == 1,
            f"DataFlow Feed call count mismatch: {label}/{sample_index}",
        )
        _require(
            _call_count(contained, "FetchDataFlowGraphTensor") == 1,
            f"DataFlow Fetch call count mismatch: {label}/{sample_index}",
        )
        _require(
            all(int(row["status"]) == 0 for row in feed + fetch),
            f"failed DataFlow record: {label}/{sample_index}",
        )

        directional = {
            "host_to_device": {"records": 0, "bytes": 0},
            "device_to_host": {"records": 0, "bytes": 0},
        }
        for row in runtime:
            direction = _runtime_direction(str(row["api"]), int(row["kind"]))
            if direction is not None and int(row["status"]) == 0:
                directional[direction]["records"] += 1
                directional[direction]["bytes"] += int(row["bytes"])
        runtime_failures = sum(int(row["status"]) != 0 for row in runtime)
        runtime_status = (
            "invalid"
            if runtime_failures
            else "observed_zero"
            if not runtime
            else "observed"
        )
        recomputed = {
            "sample": sample_index,
            "window_start_ns": window_start,
            "window_end_ns": window_end,
            "dataflow_status": "observed",
            "dataflow_feed_calls": 1,
            "dataflow_fetch_calls": 1,
            "dataflow_feed_tensor_count": len(feed),
            "dataflow_fetch_tensor_count": len(fetch),
            "dataflow_feed_tensor_bytes": feed_sizes,
            "dataflow_fetch_tensor_bytes": fetch_sizes,
            "dataflow_host_to_device_bytes": sum(feed_sizes),
            "dataflow_device_to_host_bytes": sum(fetch_sizes),
            "dataflow_total_bytes": sum(feed_sizes) + sum(fetch_sizes),
            "dataflow_failed_records": 0,
            "runtime_memcpy_status": runtime_status,
            "runtime_memcpy_records": len(runtime),
            "runtime_memcpy_failed_records": runtime_failures,
            "runtime_memcpy_host_to_device_records": directional[
                "host_to_device"
            ]["records"],
            "runtime_memcpy_device_to_host_records": directional[
                "device_to_host"
            ]["records"],
            "runtime_memcpy_host_to_device_bytes": directional[
                "host_to_device"
            ]["bytes"],
            "runtime_memcpy_device_to_host_bytes": directional[
                "device_to_host"
            ]["bytes"],
            "runtime_memcpy_total_directional_bytes": sum(
                value["bytes"] for value in directional.values()
            ),
            "mbuf_diagnostic_records": len(mbuf),
            "mbuf_diagnostic_failed_records": sum(
                int(row["status"]) != 0 for row in mbuf
            ),
        }
        reported = transfer_block["samples"][sample_index]
        for field, expected in recomputed.items():
            _require(
                reported.get(field) == expected,
                f"recomputed {field} mismatch: {label}/{sample_index}",
            )
        recomputed_samples.append(recomputed)

    _require(not boundary_call_keys, f"epoch boundary crossing: {label}")
    _require(
        transfer_block.get("boundary_crossing_records") == len(boundary_record_indexes),
        f"boundary record count mismatch: {label}",
    )
    _require(
        transfer_block.get("boundary_crossing_calls") == len(boundary_call_keys),
        f"boundary call count mismatch: {label}",
    )

    filtered_path = Path(transfer_block.get("filtered_trace_path", ""))
    _require(filtered_path.is_file(), f"missing filtered transfer trace: {label}")
    _require(
        _sha256(filtered_path) == transfer_block.get("filtered_trace_sha256"),
        f"filtered transfer trace SHA256 mismatch: {label}",
    )
    with filtered_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require(
            tuple(reader.fieldnames or ()) == FILTERED_TRACE_FIELDS,
            f"unexpected filtered transfer trace header: {label}",
        )
        actual_filtered = [
            _normalize_filtered_row(row) for row in reader
        ]
    _require(
        actual_filtered == expected_filtered,
        f"filtered transfer rows do not match raw trace: {label}",
    )

    raw_by_transport = Counter(_transport(str(call["api"])) for call in calls)
    measured_by_transport = Counter(
        _transport(str(calls[index]["api"])) for index in measured_indexes
    )
    _require(transfer_block.get("raw_record_count") == len(calls), f"raw count: {label}")
    _require(
        transfer_block.get("raw_records_by_transport")
        == dict(sorted(raw_by_transport.items())),
        f"raw transport counts: {label}",
    )
    _require(
        transfer_block.get("measured_record_count") == len(measured_indexes),
        f"measured count: {label}",
    )
    _require(
        transfer_block.get("measured_records_by_transport")
        == dict(sorted(measured_by_transport.items())),
        f"measured transport counts: {label}",
    )
    _require(
        transfer_block.get("outside_window_records_by_transport")
        == {
            transport: raw_by_transport[transport] - measured_by_transport[transport]
            for transport in sorted(raw_by_transport)
        },
        f"outside-window transport counts: {label}",
    )
    _require(
        transfer_block.get("failed_measured_records")
        == sum(int(calls[index]["status"]) != 0 for index in measured_indexes),
        f"failed measured records: {label}",
    )
    statuses = {sample["runtime_memcpy_status"] for sample in recomputed_samples}
    runtime_status = (
        "invalid"
        if "invalid" in statuses
        else "observed_zero"
        if statuses == {"observed_zero"}
        else "observed"
    )
    _require(
        transfer_block.get("runtime_memcpy_status") == runtime_status,
        f"runtime memcpy block status mismatch: {label}",
    )
    _require(transfer_block.get("status") == "observed", f"transfer block failed: {label}")
    return recomputed_samples


def _verify_route_summaries(
    transfer: dict[str, Any], route_samples: dict[str, list[dict[str, Any]]]
) -> None:
    fields = (
        "dataflow_host_to_device_bytes",
        "dataflow_device_to_host_bytes",
        "dataflow_total_bytes",
        "runtime_memcpy_records",
        "runtime_memcpy_host_to_device_bytes",
        "runtime_memcpy_device_to_host_bytes",
        "runtime_memcpy_total_directional_bytes",
    )
    for route, samples in route_samples.items():
        reported = transfer.get("routes", {}).get(route, {})
        _require(reported.get("sample_count") == 30, f"transfer route count: {route}")
        for field in fields:
            key = f"{field}_median"
            expected = statistics.median(sample[field] for sample in samples)
            _require(reported.get(key) == expected, f"transfer route median: {route}/{field}")
        _require(
            reported.get("runtime_memcpy_statuses")
            == sorted({sample["runtime_memcpy_status"] for sample in samples}),
            f"runtime status set: {route}",
        )


def validate_result(data: dict[str, Any], *, require_artifacts: bool) -> dict[str, Any]:
    _require(
        data.get("gate")
        == "Attempt 74 minimal Host-UDF ABI and control-overhead evidence",
        "unexpected gate",
    )
    _require(data.get("schema_version") == 2, "unexpected schema version")
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
    profiler_status = profiler.get("status")
    _require(profiler_status in ("observed", "not_observed"), "invalid profiler status")
    if profiler_status == "observed":
        for route in ("old", "new"):
            value = profiler.get("routes", {}).get(route, {}).get("total_bytes")
            _require(isinstance(value, int) and value >= 0, f"{route}: invalid profiler bytes")
    else:
        _require(bool(profiler.get("reason")), "not_observed has no reason")

    transfer = data.get("transfer_trace")
    _require(isinstance(transfer, dict), "missing transfer trace result")
    _require(transfer.get("schema_version") == 2, "transfer trace schema changed")
    _require(transfer.get("status") == "observed", "DataFlow payload not observed")
    _require(
        transfer.get("logical_abi_bytes_used_as_transfer") is False,
        "logical ABI bytes were substituted for observed payload",
    )
    _require(
        transfer.get("dataflow_tensor_payload_is_physical_link_bytes") is False,
        "DataFlow payload was misclaimed as physical-link bytes",
    )
    _require(
        transfer.get("physical_link_bytes_claimed") is False,
        "unsupported physical-link byte claim",
    )

    transfer_entry = data.get("transfer_trace_summary")
    _require(isinstance(transfer_entry, dict), "missing transfer trace summary input")
    if require_artifacts:
        transfer_artifact = _load_hashed(transfer_entry, "transfer_trace_summary")
        _require(transfer_artifact == transfer, "embedded transfer trace differs from input")

    transfer_blocks = transfer.get("blocks")
    _require(isinstance(transfer_blocks, dict), "missing transfer trace blocks")
    recomputed_by_label: dict[str, list[dict[str, Any]]] = {}
    for index, (label, route) in enumerate(
        zip(BLOCK_LABELS, EXPECTED_ORDER, strict=True)
    ):
        block = transfer_blocks.get(label)
        _require(isinstance(block, dict), f"missing transfer block: {label}")
        _require(block.get("route") == route, f"transfer route changed: {label}")
        _require(block.get("status") == "observed", f"transfer block failed: {label}")
        _require(
            block.get("dataflow_tensor_payload_status") == "observed",
            f"DataFlow payload failed: {label}",
        )
        _require(
            block.get("runtime_memcpy_status") in ("observed", "observed_zero"),
            f"runtime memcpy status invalid: {label}",
        )
        _require(len(block.get("samples", [])) == 15, f"transfer sample count: {label}")
        if require_artifacts:
            recomputed_by_label[label] = _recompute_transfer_block(
                label, route, block, raw_blocks[index]
            )
        else:
            recomputed_by_label[label] = block["samples"]

    transfer_route_samples = {
        "old": recomputed_by_label["old-1"] + recomputed_by_label["old-2"],
        "new": recomputed_by_label["new-1"] + recomputed_by_label["new-2"],
    }
    _verify_route_summaries(transfer, transfer_route_samples)

    observed = data.get("observed_dataflow_payload")
    _require(isinstance(observed, dict), "missing observed DataFlow payload summary")
    expected_observed = {
        "old_host_to_device_bytes_per_epoch": 58_720_516,
        "old_device_to_host_bytes_per_epoch": 78_184_928,
        "old_total_bytes_per_epoch": 136_905_444,
        "new_host_to_device_bytes_per_epoch": 260,
        "new_device_to_host_bytes_per_epoch": 368,
        "new_total_bytes_per_epoch": 628,
        "reduction_bytes_per_epoch": 136_904_816,
        "matches_declared_abi": True,
        "is_physical_link_bytes": False,
    }
    _require(observed == expected_observed, "observed DataFlow payload summary changed")

    runtime_statuses = sorted(
        {
            sample["runtime_memcpy_status"]
            for samples in transfer_route_samples.values()
            for sample in samples
        }
    )
    return {
        "pass": True,
        "raw_blocks": 4,
        "samples_per_route": 30,
        "old_dataflow_bytes_per_epoch": 136_905_444,
        "new_dataflow_bytes_per_epoch": 628,
        "dataflow_reduction_bytes_per_epoch": 136_904_816,
        "profiler_transfer_status": profiler_status,
        "runtime_memcpy_statuses": runtime_statuses,
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
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        print(f"ABI_COMPARISON_RESULT_INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
