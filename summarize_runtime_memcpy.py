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

from vllm_ascend_resident_epoch.abi import (
    ABI_BYTES,
    NEW_INPUTS,
    NEW_OUTPUTS,
    OLD_INPUTS,
    OLD_OUTPUTS,
)


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
EXPECTED_TENSOR_BYTES = {
    "old": {
        "host_to_device": [spec.nbytes for spec in OLD_INPUTS],
        "device_to_host": [spec.nbytes for spec in OLD_OUTPUTS],
    },
    "new": {
        "host_to_device": [spec.nbytes for spec in NEW_INPUTS],
        "device_to_host": [spec.nbytes for spec in NEW_OUTPUTS],
    },
}


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


def _direction(api: str, kind: int) -> str | None:
    if api in DATAFLOW_API_DIRECTIONS:
        return DATAFLOW_API_DIRECTIONS[api]
    if api in RUNTIME_MEMCPY_APIS:
        return _runtime_direction(api, kind)
    return None


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
            raise ValueError(f"unexpected transfer trace header in {path}")
        for line, row in enumerate(reader, start=2):
            try:
                parsed = {field: int(row[field]) for field in TRACE_FIELDS[1:]}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid transfer trace row {path}:{line}") from exc
            api = row.get("api")
            if api not in TRACE_APIS:
                raise ValueError(f"invalid transfer trace API {path}:{line}")
            parsed["api"] = api
            if parsed["start_ns"] <= 0 or parsed["end_ns"] < parsed["start_ns"]:
                raise ValueError(f"invalid transfer trace timestamp {path}:{line}")
            if parsed["bytes"] < 0 or parsed["dest_max"] < parsed["bytes"]:
                raise ValueError(f"invalid transfer trace byte count {path}:{line}")
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


def _dataflow_call_count(rows: list[dict[str, int | str]], api: str) -> int:
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


def _sample_summary(
    route: str,
    sample_index: int,
    window_start: int,
    window_end: int,
    rows: list[dict[str, int | str]],
) -> dict[str, Any]:
    feed_rows = [row for row in rows if row["api"] == "FeedDataFlowGraphTensor"]
    fetch_rows = [row for row in rows if row["api"] == "FetchDataFlowGraphTensor"]
    runtime_rows = [row for row in rows if row["api"] in RUNTIME_MEMCPY_APIS]
    mbuf_rows = [row for row in rows if row["api"] in MBUF_DIAGNOSTIC_APIS]

    feed_sizes = [int(row["bytes"]) for row in feed_rows]
    fetch_sizes = [int(row["bytes"]) for row in fetch_rows]
    feed_calls = _dataflow_call_count(rows, "FeedDataFlowGraphTensor")
    fetch_calls = _dataflow_call_count(rows, "FetchDataFlowGraphTensor")
    expected = EXPECTED_TENSOR_BYTES[route]
    dataflow_failures = sum(int(row["status"]) != 0 for row in feed_rows + fetch_rows)
    dataflow_observed = (
        feed_calls == 1
        and fetch_calls == 1
        and feed_sizes == expected["host_to_device"]
        and fetch_sizes == expected["device_to_host"]
        and dataflow_failures == 0
    )

    runtime_directional = {
        "host_to_device": {"records": 0, "bytes": 0},
        "device_to_host": {"records": 0, "bytes": 0},
    }
    for row in runtime_rows:
        direction = _runtime_direction(str(row["api"]), int(row["kind"]))
        if direction is not None and int(row["status"]) == 0:
            runtime_directional[direction]["records"] += 1
            runtime_directional[direction]["bytes"] += int(row["bytes"])
    runtime_failures = sum(int(row["status"]) != 0 for row in runtime_rows)
    runtime_status = (
        "invalid"
        if runtime_failures
        else "observed_zero"
        if not runtime_rows
        else "observed"
    )

    return {
        "sample": sample_index,
        "window_start_ns": window_start,
        "window_end_ns": window_end,
        "dataflow_status": "observed" if dataflow_observed else "not_observed",
        "dataflow_feed_calls": feed_calls,
        "dataflow_fetch_calls": fetch_calls,
        "dataflow_feed_tensor_count": len(feed_rows),
        "dataflow_fetch_tensor_count": len(fetch_rows),
        "dataflow_feed_tensor_bytes": feed_sizes,
        "dataflow_fetch_tensor_bytes": fetch_sizes,
        "dataflow_host_to_device_bytes": sum(feed_sizes),
        "dataflow_device_to_host_bytes": sum(fetch_sizes),
        "dataflow_total_bytes": sum(feed_sizes) + sum(fetch_sizes),
        "dataflow_failed_records": dataflow_failures,
        "runtime_memcpy_status": runtime_status,
        "runtime_memcpy_records": len(runtime_rows),
        "runtime_memcpy_failed_records": runtime_failures,
        "runtime_memcpy_host_to_device_records": runtime_directional[
            "host_to_device"
        ]["records"],
        "runtime_memcpy_device_to_host_records": runtime_directional[
            "device_to_host"
        ]["records"],
        "runtime_memcpy_host_to_device_bytes": runtime_directional[
            "host_to_device"
        ]["bytes"],
        "runtime_memcpy_device_to_host_bytes": runtime_directional[
            "device_to_host"
        ]["bytes"],
        "runtime_memcpy_total_directional_bytes": sum(
            value["bytes"] for value in runtime_directional.values()
        ),
        "mbuf_diagnostic_records": len(mbuf_rows),
        "mbuf_diagnostic_failed_records": sum(
            int(row["status"]) != 0 for row in mbuf_rows
        ),
    }


def _summarize_block(
    label: str,
    trace_path: Path,
    result_path: Path,
    filtered_path: Path,
) -> dict[str, Any]:
    route = ROUTES[label]
    calls = _read_trace(trace_path)
    windows = _load_windows(result_path)
    samples: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    measured_call_indexes: set[int] = set()
    boundary_record_indexes: set[int] = set()
    boundary_call_keys: set[tuple[Any, ...]] = set()

    for sample_index, (window_start, window_end) in enumerate(windows):
        contained_rows: list[dict[str, int | str]] = []
        for call_index, call in enumerate(calls):
            overlaps = int(call["end_ns"]) >= window_start and int(
                call["start_ns"]
            ) <= window_end
            contained = int(call["start_ns"]) >= window_start and int(
                call["end_ns"]
            ) <= window_end
            if overlaps and not contained:
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
            if not contained:
                continue
            measured_call_indexes.add(call_index)
            contained_rows.append(call)
            api = str(call["api"])
            filtered_rows.append(
                {
                    "block": label,
                    "sample": sample_index,
                    "transport": _transport(api),
                    "direction": _direction(api, int(call["kind"])) or "",
                    **call,
                }
            )
        samples.append(
            _sample_summary(
                route,
                sample_index,
                window_start,
                window_end,
                contained_rows,
            )
        )

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with filtered_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=FILTERED_TRACE_FIELDS, delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(filtered_rows)

    raw_by_transport = Counter(_transport(str(call["api"])) for call in calls)
    measured_by_transport = Counter(
        _transport(str(calls[index]["api"])) for index in measured_call_indexes
    )
    failed_measured_records = sum(
        int(calls[index]["status"]) != 0 for index in measured_call_indexes
    )
    dataflow_observed = all(
        sample["dataflow_status"] == "observed" for sample in samples
    )
    status = (
        "observed"
        if dataflow_observed
        and not boundary_call_keys
        and failed_measured_records == 0
        else "not_observed"
    )
    runtime_statuses = {sample["runtime_memcpy_status"] for sample in samples}
    runtime_status = (
        "invalid"
        if "invalid" in runtime_statuses or boundary_call_keys
        else "observed_zero"
        if runtime_statuses == {"observed_zero"}
        else "observed"
    )
    return {
        "route": route,
        "status": status,
        "reason": None
        if status == "observed"
        else (
            "every measured epoch must contain one complete DataFlow Feed/Fetch "
            "tensor payload matching its ABI, with no failed or boundary-crossing calls"
        ),
        "dataflow_tensor_payload_status": (
            "observed" if dataflow_observed else "not_observed"
        ),
        "runtime_memcpy_status": runtime_status,
        "trace_path": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "benchmark_result_path": str(result_path),
        "benchmark_result_sha256": _sha256(result_path),
        "filtered_trace_path": str(filtered_path),
        "filtered_trace_sha256": _sha256(filtered_path),
        "raw_record_count": len(calls),
        "raw_records_by_transport": dict(sorted(raw_by_transport.items())),
        "measured_record_count": len(measured_call_indexes),
        "measured_records_by_transport": dict(
            sorted(measured_by_transport.items())
        ),
        "outside_window_records_by_transport": {
            transport: raw_by_transport[transport] - measured_by_transport[transport]
            for transport in sorted(raw_by_transport)
        },
        "failed_measured_records": failed_measured_records,
        "boundary_crossing_records": len(boundary_record_indexes),
        "boundary_crossing_calls": len(boundary_call_keys),
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
            "dataflow_host_to_device_bytes_median": statistics.median(
                sample["dataflow_host_to_device_bytes"] for sample in samples
            ),
            "dataflow_device_to_host_bytes_median": statistics.median(
                sample["dataflow_device_to_host_bytes"] for sample in samples
            ),
            "dataflow_total_bytes_median": statistics.median(
                sample["dataflow_total_bytes"] for sample in samples
            ),
            "runtime_memcpy_records_median": statistics.median(
                sample["runtime_memcpy_records"] for sample in samples
            ),
            "runtime_memcpy_host_to_device_bytes_median": statistics.median(
                sample["runtime_memcpy_host_to_device_bytes"] for sample in samples
            ),
            "runtime_memcpy_device_to_host_bytes_median": statistics.median(
                sample["runtime_memcpy_device_to_host_bytes"] for sample in samples
            ),
            "runtime_memcpy_total_directional_bytes_median": statistics.median(
                sample["runtime_memcpy_total_directional_bytes"] for sample in samples
            ),
            "runtime_memcpy_statuses": sorted(
                {sample["runtime_memcpy_status"] for sample in samples}
            ),
        }
    return {
        "schema_version": 2,
        "metric": "DataFlow tensor payload and CANN runtime memcpy activity inside measured EngineCore epoch windows",
        "measurement_boundaries": {
            "dataflow_tensor": (
                "Tensor::GetSize() on the actual DFlowSessionImpl FeedDataFlowGraph and "
                "FetchDataFlowGraph arguments"
            ),
            "runtime_memcpy": (
                "LD_PRELOAD interception of CANN rtMemcpy/rtsMemcpy API families in the "
                "resident sidecar"
            ),
            "mbuf_diagnostic": (
                "LD_PRELOAD interception of selected Mbuf/Buff allocation and size APIs"
            ),
        },
        "status": status,
        "reason": None
        if status == "observed"
        else "; ".join(
            f"{label}: {block['reason']}"
            for label, block in blocks.items()
            if block["status"] != "observed"
        ),
        "logical_abi_bytes_used_as_transfer": False,
        "dataflow_tensor_payload_is_physical_link_bytes": False,
        "physical_link_bytes_claimed": False,
        "clock": "CLOCK_REALTIME/time.time_ns",
        "blocks": blocks,
        "routes": route_summary,
        "expected_dataflow_payload": {
            route: {
                "host_to_device_tensor_bytes": values["host_to_device"],
                "device_to_host_tensor_bytes": values["device_to_host"],
                "host_to_device_bytes": ABI_BYTES[route]["input"],
                "device_to_host_bytes": ABI_BYTES[route]["output"],
            }
            for route, values in EXPECTED_TENSOR_BYTES.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for label in BLOCKS:
        option = label.replace("-", "_")
        parser.add_argument(
            f"--{label}-trace", dest=f"{option}_trace", type=Path, required=True
        )
        parser.add_argument(
            f"--{label}-result", dest=f"{option}_result", type=Path, required=True
        )
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
        print(f"TRANSFER_TRACE_SUMMARY_INVALID: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
