#!/usr/bin/env python3
"""Verify the P2 cancellation mechanism and profiler attribution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import re
from pathlib import Path
from typing import Any


FORMAT = "cruise-persistent-cancel-p2-v1"
PLACEMENT_FORMAT = "cruise-persistent-cancel-p2-placement-v1"
TARGET_PROCESS_POINT = "persistent_cancel_p2_pp"
TARGET_FLOW_FUNC = "persistent_cancel_p2"
TARGET_OP = "persistent_cancel_p2_add"
EXPECTED_FEEDS = 2028
EXPECTED_ADMISSIONS = 1009
EXPECTED_CANCELLED = 1008
EXPECTED_RETIRED = 1009
EXPECTED_DUPLICATES = 4
EXPECTED_REJECTED = 5
EXPECTED_QUIESCENT = 255
EXPECTED_LATENCY_SAMPLES = 1000
MAX_PROFILE_FILE_RECORDS = 128
MAX_CONTROLLER_EVIDENCE_LINES = 128

OUTPUT_ADMIT_ACK = 1
OUTPUT_COMMIT = 2
OUTPUT_RETIRE_COMPLETE = 3
OUTPUT_RETIRE_CANCELLED = 4
OUTPUT_CREDIT_ACK = 5
OUTPUT_QUIESCENT = 6
OUTPUT_CUMULATIVE_ACK = 7
OUTPUT_REJECTED = 8
OUTPUT_SHUTDOWN = 9

_OWNER_START = re.compile(r"^P2_OWNER_START pid=(\d+)\s")
_UDF_EXECUTOR = re.compile(r"\bUDF\((\d+),udf_executor\):")
_CONTROLLER_MODEL_INIT = re.compile(
    rf"parse name={TARGET_PROCESS_POINT} end, "
    rf"flowFuncName={TARGET_FLOW_FUNC}, "
    rf"instanceName={TARGET_PROCESS_POINT}@"
)
_CONTROLLER_PROCESSOR_INIT = re.compile(
    rf"end to init FlowFunc processor, "
    rf"flow_func_info={TARGET_FLOW_FUNC}\[{TARGET_PROCESS_POINT}@"
)
_CONTROLLER_SUMMARY = re.compile(
    rf"flow_func_info={TARGET_FLOW_FUNC}\[{TARGET_PROCESS_POINT}@[^]]*\].*"
    r"call flow func times=(\d+), schedule finish times=(\d+), "
    r"set output times=\[(\d+)\]"
)
_CONTROLLER_METRICS = re.compile(
    rf"model_metrics:name={TARGET_FLOW_FUNC}\[{TARGET_PROCESS_POINT}@[^]]*\], "
    r"min_exec_time=(\d+) us, max_exec_time=(\d+) us, "
    r"sub_max_exec_time=(\d+) us, total_exec_time=(\d+) us, "
    r"total_exec_num=(\d+)"
)
_CONTROLLER_EXIT = re.compile(r"\bFlow func executor exit\.")


def parse_fields(line: str) -> dict[str, int]:
    fields: dict[str, int] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = int(value)
    return fields


def parse_usage_percent(path: Path) -> int:
    match = re.search(
        r"HBM Usage Rate\(%\)\s*:\s*(\d+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    if match is None:
        raise ValueError(f"cannot parse HBM usage from {path}")
    return int(match.group(1))


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _is_ai_core_task(value: str) -> bool:
    normalized = _normalize(value).upper()
    if "AICPU" in normalized or "AI_CPU" in normalized:
        return False
    return normalized in {
        "AIC",
        "AIV",
        "AI_CORE",
        "AI_VECTOR_CORE",
        "AICORE",
        "AIVECTORCORE",
    }


def _is_ai_cpu_task(value: str) -> bool:
    normalized = _normalize(value).upper()
    return "AICPU" in normalized or "AI_CPU" in normalized


def _is_target_add(name: str) -> bool:
    normalized = _normalize(name)
    return normalized == TARGET_OP or normalized.endswith(f"_{TARGET_OP}")


def _parse_profile_status(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) != 2 or not row[0] or row[0] in values:
                raise ValueError(f"invalid profile status row in {path}: {row}")
            values[row[0]] = row[1]
    return values


def _relative_path(path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return path.name


def expected_requests() -> dict[tuple[int, int], tuple[int, int, int]]:
    values = {
        (101, 1): (1000, 8, 0),
        (102, 2): (2000, 4, 4),
        (103, 3): (3000, 256, 256),
        (104, 4): (4000, 64, 16),
        (201, 10): (5000, 256, 256),
        (202, 11): (5100, 256, 256),
        (203, 12): (5200, 256, 256),
        (204, 13): (5300, 256, 256),
        (205, 14): (6000, 256, 256),
    }
    for index in range(EXPECTED_LATENCY_SAMPLES):
        generation = 1000 + index
        values[(10000 + index, generation)] = (
            100000 + generation,
            256,
            256,
        )
    return values


def _percentile(values: list[int], percent: int) -> int:
    ordered = sorted(values)
    rank = (percent * len(ordered) + 99) // 100
    return ordered[rank - 1]


def verify_start(path: Path) -> dict[str, Any]:
    owner_starts: list[dict[str, int]] = []
    outputs: list[dict[str, int]] = []
    matrices: list[dict[str, int]] = []
    latencies: list[dict[str, int]] = []
    summaries: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("P2_OWNER_START "):
            owner_starts.append(parse_fields(line))
        elif line.startswith("P2_OUTPUT "):
            outputs.append(parse_fields(line))
        elif line.startswith("P2_MATRIX "):
            matrices.append(parse_fields(line))
        elif line.startswith("P2_CANCEL_LATENCY "):
            latencies.append(parse_fields(line))
        elif line.startswith("P2_SUMMARY "):
            summaries.append(parse_fields(line))

    errors: list[str] = []
    if len(owner_starts) != 1:
        errors.append(f"owner starts={len(owner_starts)}, expected 1")
    if len(matrices) != 1:
        errors.append(f"matrix summaries={len(matrices)}, expected 1")
    if len(latencies) != 1:
        errors.append(f"latency summaries={len(latencies)}, expected 1")
    if len(summaries) != 1:
        errors.append(f"run summaries={len(summaries)}, expected 1")
    owner = summaries[0].get("owner") if len(summaries) == 1 else None
    if owner is not None and (
        len(owner_starts) != 1 or owner_starts[0].get("owner") != owner
    ):
        errors.append("Owner Instance changed within the run")
    if owner is not None and any(output.get("owner") != owner for output in outputs):
        errors.append("an output belongs to a different Owner Instance")

    expected = expected_requests()
    admitted: dict[tuple[int, int], dict[str, int]] = {}
    commits: dict[tuple[int, int], list[tuple[int, dict[str, int]]]] = defaultdict(list)
    retirements: dict[tuple[int, int], list[tuple[int, dict[str, int]]]] = defaultdict(list)
    active_rows: dict[int, tuple[int, int]] = {}
    last_generation = [-1, -1, -1, -1]
    type_counts = Counter(output.get("type") for output in outputs)
    final_shutdown: dict[str, int] | None = None

    for index, output in enumerate(outputs):
        output_type = output.get("type")
        key = (output.get("request", 0), output.get("generation", 0))
        row = output.get("row", -1)
        if output.get("transaction") != output.get("event_seq"):
            errors.append(f"output[{index}] transaction does not match event")
            break
        if output_type == OUTPUT_ADMIT_ACK:
            if key not in expected:
                errors.append(f"output[{index}] admitted unexpected identity={key}")
                break
            if key in admitted:
                errors.append(f"output[{index}] repeated admission effect={key}")
                break
            if row not in range(4) or row in active_rows:
                errors.append(f"output[{index}] reused active row={row}")
                break
            if output.get("commit_seq") != 0 or output.get("status") != 0:
                errors.append(f"output[{index}] invalid Admit Ack={output}")
                break
            if output.get("generation", 0) <= last_generation[row]:
                errors.append(f"output[{index}] generation did not advance on row={row}")
                break
            last_generation[row] = output["generation"]
            active_rows[row] = key
            admitted[key] = {"row": row, "index": index}
        elif output_type == OUTPUT_COMMIT:
            if row not in active_rows or active_rows[row] != key:
                errors.append(f"output[{index}] commit is not owned by active row={row}")
                break
            commits[key].append((index, output))
        elif output_type in (OUTPUT_RETIRE_COMPLETE, OUTPUT_RETIRE_CANCELLED):
            if row not in active_rows or active_rows[row] != key:
                errors.append(f"output[{index}] retirement has no active owner={key}")
                break
            retirements[key].append((index, output))
            del active_rows[row]
        elif output_type == OUTPUT_QUIESCENT and active_rows:
            errors.append(f"output[{index}] quiescent with active rows={active_rows}")
            break
        elif output_type == OUTPUT_SHUTDOWN:
            if active_rows:
                errors.append(f"output[{index}] shutdown with active rows={active_rows}")
                break
            final_shutdown = output

    if set(admitted) != set(expected):
        errors.append(
            f"admitted identities differ: missing={len(set(expected) - set(admitted))} "
            f"extra={len(set(admitted) - set(expected))}"
        )
    for key, (seed, target, initial_credit) in expected.items():
        records = commits.get(key, [])
        sequences = [record.get("commit_seq") for _, record in records]
        if sequences != list(range(1, len(records) + 1)):
            errors.append(f"identity={key} commit sequence is not contiguous")
            continue
        for _, record in records:
            commit = record["commit_seq"]
            expected_credit = initial_credit - commit
            if key == (104, 4) and commit > 16:
                expected_credit += 48
            if (
                record.get("state") != seed + commit
                or record.get("remaining") != target - commit
                or record.get("credit") != expected_credit
                or record.get("status") != 0
            ):
                errors.append(f"identity={key} invalid commit={record}")
                break
        retired = retirements.get(key, [])
        if len(retired) != 1:
            errors.append(f"identity={key} retirements={len(retired)}, expected 1")
            continue
        retire_index, retirement = retired[0]
        if retirement.get("commit_seq") != len(records):
            errors.append(f"identity={key} retirement prefix disagrees with commits")
        if any(index > retire_index for index, _ in records):
            errors.append(f"identity={key} committed after retirement")
        completed = key == (102, 2)
        expected_type = OUTPUT_RETIRE_COMPLETE if completed else OUTPUT_RETIRE_CANCELLED
        if retirement.get("type") != expected_type:
            errors.append(f"identity={key} retirement type={retirement.get('type')}")
        if completed and len(records) != target:
            errors.append(f"completed identity={key} commits={len(records)}")
        if not completed and len(records) >= target:
            errors.append(f"cancelled identity={key} reached target={target}")

    expected_types = {
        OUTPUT_ADMIT_ACK: EXPECTED_ADMISSIONS,
        OUTPUT_RETIRE_COMPLETE: 1,
        OUTPUT_RETIRE_CANCELLED: EXPECTED_CANCELLED,
        OUTPUT_CREDIT_ACK: 1,
        OUTPUT_QUIESCENT: EXPECTED_QUIESCENT,
        OUTPUT_CUMULATIVE_ACK: EXPECTED_DUPLICATES,
        OUTPUT_REJECTED: EXPECTED_REJECTED,
        OUTPUT_SHUTDOWN: 1,
    }
    for output_type, count in expected_types.items():
        if type_counts[output_type] != count:
            errors.append(
                f"output type={output_type} count={type_counts[output_type]}, expected={count}"
            )
    duplicate_sequences = sorted(
        output["event_seq"]
        for output in outputs
        if output.get("type") == OUTPUT_CUMULATIVE_ACK
    )
    if duplicate_sequences != [1, 2, 7, 10]:
        errors.append(f"duplicate event sequences={duplicate_sequences}")
    expected_rejections = {(4, 9), (3, 2), (5, 1), (9, 4), (16, 3)}
    actual_rejections = {
        (output["event_seq"], output["status"])
        for output in outputs
        if output.get("type") == OUTPUT_REJECTED
    }
    if actual_rejections != expected_rejections:
        errors.append(f"rejections={actual_rejections}, expected={expected_rejections}")

    stress_latencies = [
        output["host_cancel_latency_us"]
        for output in outputs
        if output.get("type") == OUTPUT_RETIRE_CANCELLED
        and output.get("request", 0) >= 10000
    ]
    if len(stress_latencies) != EXPECTED_LATENCY_SAMPLES or any(
        value < 0 for value in stress_latencies
    ):
        errors.append(
            f"valid stress latency samples={len(stress_latencies)}, expected=1000"
        )
    calculated_p95 = _percentile(stress_latencies, 95) if stress_latencies else -1
    calculated_p99 = _percentile(stress_latencies, 99) if stress_latencies else -1
    if calculated_p95 > 50000 or calculated_p99 > 100000:
        errors.append(
            f"cancel latency p95={calculated_p95}us p99={calculated_p99}us"
        )

    matrix = matrices[0] if len(matrices) == 1 else {}
    expected_matrix = {
        "before_first_commits": 0,
        "completed_commits": 4,
        "blocked_commits": 16,
        "duplicate_acks": EXPECTED_DUPLICATES,
        "rejected": EXPECTED_REJECTED,
        "matrix_cancelled": 8,
        "retirement_before_reuse": 1,
        "stale_generation_isolated": 1,
    }
    for key, value in expected_matrix.items():
        if matrix.get(key) != value:
            errors.append(f"matrix {key}={matrix.get(key)}, expected={value}")
    if not 32 <= matrix.get("during_commits", -1) < 256:
        errors.append(f"matrix during_commits={matrix.get('during_commits')}")

    latency = latencies[0] if len(latencies) == 1 else {}
    if (
        latency.get("samples") != EXPECTED_LATENCY_SAMPLES
        or latency.get("p95_us") != calculated_p95
        or latency.get("p99_us") != calculated_p99
    ):
        errors.append("Host latency summary disagrees with retirement records")

    summary = summaries[0] if len(summaries) == 1 else {}
    expected_summary = {
        "owner_starts": 1,
        "feed_calls": EXPECTED_FEEDS,
        "matrix_admissions": 9,
        "matrix_cancelled": 8,
        "stress_admissions": 1000,
        "stress_cancelled": 1000,
        "duplicate_acks": EXPECTED_DUPLICATES,
        "rejected": EXPECTED_REJECTED,
        "quiescent": EXPECTED_QUIESCENT,
        "total_retired": EXPECTED_RETIRED,
        "latency_samples": EXPECTED_LATENCY_SAMPLES,
        "latency_p95_us": calculated_p95,
        "latency_p99_us": calculated_p99,
        "compile_status": 0,
        "fetch_status": 0,
        "remove_status": 0,
        "finalize_status": 0,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            errors.append(f"summary {key}={summary.get(key)}, expected={value}")
    if summary.get("fetch_calls") != len(outputs):
        errors.append(
            f"summary fetch_calls={summary.get('fetch_calls')}, outputs={len(outputs)}"
        )
    if summary.get("total_commits") != sum(len(value) for value in commits.values()):
        errors.append("summary total commits disagrees with output sequence")
    if summary.get("aicore_calls", 0) <= 0:
        errors.append("summary contains no AICore recurrence calls")
    if summary.get("host_cpu_us", 0) <= 0 or summary.get("wall_ms", 0) <= 0:
        errors.append("Host lifecycle timing is not positive")
    if final_shutdown is None:
        errors.append("shutdown output is missing")
    elif (
        final_shutdown.get("aicore_calls") != summary.get("aicore_calls")
        or final_shutdown.get("total_commits") != summary.get("total_commits")
        or final_shutdown.get("total_retired") != EXPECTED_RETIRED
        or final_shutdown.get("quiescent") != EXPECTED_QUIESCENT
        or final_shutdown.get("flags") != 1
    ):
        errors.append("shutdown counters disagree with the run summary")
    return {
        "path": str(path),
        "pass": not errors,
        "errors": errors,
        "summary": summary,
        "matrix": matrix,
        "latency": {
            "samples": len(stress_latencies),
            "p95_us": calculated_p95,
            "p99_us": calculated_p99,
        },
        "outputs": len(outputs),
        "type_counts": dict(sorted(type_counts.items())),
    }


def analyze_controller_activity(
    profile_log: Path | None,
    log_roots: list[Path],
    expected_outputs: int,
) -> dict[str, Any]:
    errors: list[str] = []
    host_pids: list[int] = []
    if profile_log is not None and profile_log.is_file():
        for line in profile_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            match = _OWNER_START.match(line)
            if match is not None:
                host_pids.append(int(match.group(1)))
    host_pid = host_pids[0] if len(host_pids) == 1 else None
    if host_pid is None:
        errors.append(f"profile Host process ids={host_pids}, expected exactly one")

    resolved_roots = [root.resolve(strict=True) for root in log_roots]
    files: set[Path] = set()
    if host_pid is not None:
        prefix = f"device-{host_pid}_"
        for root in resolved_roots:
            if root.is_file():
                if root.name.startswith(prefix) and root.suffix == ".log":
                    files.add(root)
            else:
                files.update(
                    path.resolve()
                    for path in root.rglob(f"{prefix}*.log")
                    if path.is_file()
                )
    if not files:
        errors.append("no Device log is bound to the profiled Host process")

    records: dict[str, list[dict[str, Any]]] = {
        "model_init": [],
        "processor_init": [],
        "summary": [],
        "metrics": [],
        "executor_exit": [],
    }
    evidence_lines: list[str] = []
    bytes_scanned = 0
    for path in sorted(files):
        bytes_scanned += path.stat().st_size
        display_path = _relative_path(path, resolved_roots)
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = raw_line.rstrip("\r\n")
                executor = _UDF_EXECUTOR.search(line)
                if executor is None:
                    continue
                kind: str | None = None
                values: dict[str, int] = {}
                if _CONTROLLER_MODEL_INIT.search(line) is not None:
                    kind = "model_init"
                elif _CONTROLLER_PROCESSOR_INIT.search(line) is not None:
                    kind = "processor_init"
                else:
                    summary = _CONTROLLER_SUMMARY.search(line)
                    metrics = _CONTROLLER_METRICS.search(line)
                    if summary is not None:
                        kind = "summary"
                        values = {
                            "call_flow_func_times": int(summary.group(1)),
                            "schedule_finish_times": int(summary.group(2)),
                            "set_output_times": int(summary.group(3)),
                        }
                    elif metrics is not None:
                        kind = "metrics"
                        values = {
                            "min_exec_time_us": int(metrics.group(1)),
                            "max_exec_time_us": int(metrics.group(2)),
                            "sub_max_exec_time_us": int(metrics.group(3)),
                            "total_exec_time_us": int(metrics.group(4)),
                            "total_exec_num": int(metrics.group(5)),
                        }
                    elif _CONTROLLER_EXIT.search(line) is not None:
                        kind = "executor_exit"
                if kind is None:
                    continue
                record: dict[str, Any] = {
                    "executor_pid": int(executor.group(1)),
                    "path": display_path,
                    "line": line_number,
                }
                record.update(values)
                records[kind].append(record)
                if len(evidence_lines) < MAX_CONTROLLER_EVIDENCE_LINES:
                    evidence_lines.append(f"{display_path}:{line_number}:{line}")

    executor_pids = sorted(
        {
            record["executor_pid"]
            for entries in records.values()
            for record in entries
        }
    )
    if len(executor_pids) != 1:
        errors.append(
            f"target Device UDF executor pids={executor_pids}, expected exactly one"
        )
    for kind, entries in records.items():
        if len(entries) != 1:
            errors.append(f"controller {kind} records={len(entries)}, expected 1")
    if all(len(entries) == 1 for entries in records.values()):
        ordered = [
            records[kind][0]
            for kind in (
                "model_init",
                "processor_init",
                "summary",
                "metrics",
                "executor_exit",
            )
        ]
        if len({record["path"] for record in ordered}) != 1 or [
            record["line"] for record in ordered
        ] != sorted(record["line"] for record in ordered):
            errors.append("controller lifecycle records are not ordered in one Device log")
        summary = records["summary"][0]
        metrics = records["metrics"][0]
        if summary["call_flow_func_times"] < 1:
            errors.append("controller FlowFunc was never called")
        if summary["schedule_finish_times"] != 1:
            errors.append("controller did not finish exactly one schedule")
        if summary["set_output_times"] != expected_outputs:
            errors.append(
                f"Device controller outputs={summary['set_output_times']}, "
                f"profile Host parsed outputs={expected_outputs}"
            )
        if metrics["total_exec_num"] != 1 or metrics["total_exec_time_us"] <= 0:
            errors.append("controller execution metrics are not one positive run")
    return {
        "pass": not errors,
        "errors": errors,
        "profile_host_pid": host_pid,
        "device_executor_pids": executor_pids,
        "files_scanned": [
            _relative_path(path, resolved_roots) for path in sorted(files)
        ],
        "bytes_scanned": bytes_scanned,
        "expected_outputs": expected_outputs,
        "records": records,
        "evidence_lines": evidence_lines,
    }


def analyze_profile(
    profile_root: Path,
    profile_status: Path,
    controller: dict[str, Any],
    expected_aicore_calls: int,
) -> dict[str, Any]:
    files = sorted(path for path in profile_root.rglob("*") if path.is_file())
    sources: list[dict[str, Any]] = []
    lifecycle_types: set[str] = set()
    for path in files:
        if (
            not path.name.startswith(("task_time_", "op_summary_"))
            or path.suffix != ".csv"
        ):
            continue
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
            reader = csv.DictReader(stream)
            columns = {_normalize(name): name for name in (reader.fieldnames or [])}
            type_column = next(
                (
                    columns[name]
                    for name in ("kernel_type", "task_type", "kernel_task_type")
                    if name in columns
                ),
                None,
            )
            name_column = next(
                (
                    columns[name]
                    for name in ("kernel_name", "op_name", "task_name")
                    if name in columns
                ),
                None,
            )
            if type_column is None:
                continue
            rows = [
                (row.get(type_column, ""), row.get(name_column, "") if name_column else "")
                for row in reader
            ]
        lifecycle_types.update(_normalize(task_type).upper() for task_type, _ in rows)
        sources.append(
            {
                "path": str(path.relative_to(profile_root)),
                "rows": len(rows),
                "ai_core_tasks": sum(
                    1 for task_type, _ in rows if _is_ai_core_task(task_type)
                ),
                "ai_cpu_tasks": sum(
                    1 for task_type, _ in rows if _is_ai_cpu_task(task_type)
                ),
                "target_add_tasks": sum(
                    1
                    for task_type, name in rows
                    if _is_ai_core_task(task_type) and _is_target_add(name)
                ),
                "task_types": dict(sorted(Counter(value for value, _ in rows).items())),
                "task_names": dict(sorted(Counter(value for _, value in rows if value).items())),
            }
        )
    status_error: str | None = None
    try:
        status = _parse_profile_status(profile_status)
    except ValueError as error:
        status = {}
        status_error = str(error)
    expected_status = {
        "profile_exit_status": "0",
        "aicpu": "on",
        "ai_core": "on",
        "task_time": "l1",
    }
    ai_core_count = max((source["ai_core_tasks"] for source in sources), default=0)
    ai_cpu_count = max((source["ai_cpu_tasks"] for source in sources), default=0)
    target_count = max((source["target_add_tasks"] for source in sources), default=0)
    capture_started = "PROFILING_ENABLE" in lifecycle_types
    capture_stopped = "PROFILING_DISABLE" in lifecycle_types
    errors: list[str] = []
    if status != expected_status:
        errors.append(f"profile command status={status}, expected={expected_status}")
    if not files or not sources:
        errors.append("profile contains no readable task evidence")
    if not capture_started or not capture_stopped:
        errors.append("profile capture lifecycle is incomplete")
    if ai_core_count != expected_aicore_calls:
        errors.append(
            f"AICore/AIVector tasks={ai_core_count}, expected={expected_aicore_calls}"
        )
    if target_count != expected_aicore_calls:
        errors.append(
            f"attributed target Add tasks={target_count}, expected={expected_aicore_calls}"
        )
    if controller.get("pass") is not True:
        errors.append("profile controller lifecycle attribution failed")
    return {
        "pass": not errors,
        "errors": errors,
        "command_pass": status == expected_status,
        "profile_status": status,
        "profile_status_error": status_error,
        "file_count": len(files),
        "file_bytes": sum(path.stat().st_size for path in files),
        "files": [str(path.relative_to(profile_root)) for path in files[:MAX_PROFILE_FILE_RECORDS]],
        "files_truncated": len(files) > MAX_PROFILE_FILE_RECORDS,
        "task_sources": sources,
        "capture_started": capture_started,
        "capture_stopped": capture_stopped,
        "ai_cpu_task_count": ai_cpu_count,
        "ai_core_task_count": ai_core_count,
        "target_add_task_count": target_count,
        "expected_target_add_task_count": expected_aicore_calls,
        "target_op": TARGET_OP,
        "controller": controller,
    }


def analyze(
    run_logs: list[Path],
    npu_before: Path,
    npu_after: list[Path],
    profile_status: Path,
    profile_root: Path,
    placement: dict[str, Any],
    *,
    profile_log: Path | None = None,
    controller_log_roots: list[Path] | None = None,
    npu_after_profile: Path | None = None,
) -> dict[str, Any]:
    starts = [verify_start(path) for path in run_logs]
    owners = [start.get("summary", {}).get("owner") for start in starts]
    before_hbm = parse_usage_percent(npu_before)
    after_hbm = [parse_usage_percent(path) for path in npu_after]
    hbm_pass = len(after_hbm) == 3 and all(
        value <= before_hbm + 2 for value in after_hbm
    )
    unique_owners = len(set(owners)) == 3 and None not in owners
    mechanism_pass = (
        placement.get("format") == PLACEMENT_FORMAT
        and placement.get("pass") is True
        and len(starts) == 3
        and all(start["pass"] for start in starts)
        and unique_owners
        and hbm_pass
    )
    profile_run = (
        verify_start(profile_log)
        if profile_log is not None
        else {
            "path": None,
            "pass": False,
            "errors": ["profile mechanism log was not provided"],
            "outputs": 0,
            "summary": {},
        }
    )
    controller = analyze_controller_activity(
        profile_log,
        controller_log_roots or [],
        profile_run.get("outputs", 0),
    )
    expected_calls = profile_run.get("summary", {}).get("aicore_calls", 0)
    profile = analyze_profile(
        profile_root, profile_status, controller, expected_calls
    )
    profile_owner = profile_run.get("summary", {}).get("owner")
    profile_owner_unique = profile_owner is not None and profile_owner not in owners
    profile_hbm = (
        parse_usage_percent(npu_after_profile)
        if npu_after_profile is not None
        else None
    )
    profile_hbm_pass = profile_hbm is not None and profile_hbm <= before_hbm + 2
    profile_pass = (
        profile["pass"]
        and profile_run["pass"]
        and profile_owner_unique
        and profile_hbm_pass
    )
    profile["pass"] = profile_pass
    profile["run"] = profile_run
    profile["owner_unique_from_mechanism_runs"] = profile_owner_unique
    profile["hbm_usage_percent"] = {
        "after_profile": profile_hbm,
        "pass": profile_hbm_pass,
        "allowed_growth_points": 2,
    }
    return {
        "format": FORMAT,
        "mechanism_pass": mechanism_pass,
        "profile_pass": profile_pass,
        "pass": mechanism_pass and profile_pass,
        "owner_instances_unique": unique_owners,
        "placement": placement,
        "starts": starts,
        "hbm_usage_percent": {
            "before": before_hbm,
            "after_each_start": after_hbm,
            "pass": hbm_pass,
            "allowed_growth_points": 2,
        },
        "profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", type=Path, action="append", required=True)
    parser.add_argument("--npu-before", type=Path, required=True)
    parser.add_argument("--npu-after", type=Path, action="append", required=True)
    parser.add_argument("--profile-status", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument("--controller-log-root", type=Path, action="append")
    parser.add_argument("--controller-extract-output", type=Path)
    parser.add_argument("--npu-after-profile", type=Path)
    parser.add_argument("--placement-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanism-only", action="store_true")
    args = parser.parse_args()
    placement = json.loads(args.placement_result.read_text(encoding="utf-8"))
    result = analyze(
        args.run_log,
        args.npu_before,
        args.npu_after,
        args.profile_status,
        args.profile_root,
        placement,
        profile_log=args.profile_log,
        controller_log_roots=args.controller_log_root,
        npu_after_profile=args.npu_after_profile,
    )
    result["requested_gate"] = "mechanism" if args.mechanism_only else "full_p2"
    result["requested_gate_pass"] = (
        result["mechanism_pass"] if args.mechanism_only else result["pass"]
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.controller_extract_output is not None:
        lines = result["profile"]["controller"]["evidence_lines"]
        args.controller_extract_output.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["requested_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
