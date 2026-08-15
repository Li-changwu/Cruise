#!/usr/bin/env python3
"""Verify the P1 mechanism and its target-hardware profiler evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


FORMAT = "cruise-persistent-recurrence-p1-v1"
PLACEMENT_FORMAT = "cruise-persistent-recurrence-p1-placement-v1"
TARGET_PROCESS_POINT = "persistent_recurrence_p1_pp"
TARGET_FLOW_FUNC = "persistent_recurrence_p1"
TARGET_OP = "persistent_recurrence_p1_add"
EXPECTED_AICORE_CALLS = 752
EXPECTED_COMMITS = 1280
EXPECTED_OUTPUTS = 1291
EXPECTED_PRE_SHUTDOWN_OUTPUTS = 1290
MAX_PROFILE_FILE_RECORDS = 128
MAX_CONTROLLER_EVIDENCE_LINES = 128


_OWNER_START = re.compile(r"^P1_OWNER_START pid=(\d+)\s")
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


def _is_target_add(name: str, _op_type: str = "") -> bool:
    normalized_name = _normalize(name)
    return (
        normalized_name == TARGET_OP
        or normalized_name.endswith(f"_{TARGET_OP}")
    )


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
            errors.append(
                f"controller schedule finishes={summary['schedule_finish_times']}, "
                "expected 1"
            )
        if summary["set_output_times"] != expected_outputs:
            errors.append(
                f"Device controller outputs={summary['set_output_times']}, "
                f"profile Host parsed outputs={expected_outputs}"
            )
        if metrics["total_exec_num"] != 1 or metrics["total_exec_time_us"] <= 0:
            errors.append(
                "controller execution metrics do not prove one positive-duration run"
            )

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
        "evidence_lines_truncated": sum(len(value) for value in records.values())
        > len(evidence_lines),
    }


def _profile_source(
    path: Path,
    profile_root: Path,
    rows: list[tuple[str, str]],
    kind: str,
) -> dict[str, Any]:
    type_counts = Counter(task_type for task_type, _ in rows)
    name_counts = Counter(name for _, name in rows if name)
    ai_core = sum(
        1 for task_type, _ in rows if _is_ai_core_task(task_type)
    )
    ai_cpu = sum(1 for task_type, _ in rows if _is_ai_cpu_task(task_type))
    target_add = sum(
        1
        for task_type, name in rows
        if _is_ai_core_task(task_type) and _is_target_add(name)
    )
    return {
        "path": str(path.relative_to(profile_root)),
        "format": kind,
        "rows": len(rows),
        "ai_core_tasks": ai_core,
        "ai_cpu_tasks": ai_cpu,
        "target_add_tasks": target_add,
        "task_types": dict(sorted(type_counts.items())),
        "task_names": dict(sorted(name_counts.items())),
    }


def analyze_profile(
    profile_root: Path,
    profile_status: Path,
    controller: dict[str, Any],
) -> dict[str, Any]:
    files = sorted(path for path in profile_root.rglob("*") if path.is_file())
    sources: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
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
                    columns[candidate]
                    for candidate in ("kernel_type", "task_type", "kernel_task_type")
                    if candidate in columns
                ),
                None,
            )
            name_column = next(
                (
                    columns[candidate]
                    for candidate in ("kernel_name", "op_name", "task_name")
                    if candidate in columns
                ),
                None,
            )
            schemas.append(
                {
                    "path": str(path.relative_to(profile_root)),
                    "format": "csv",
                    "type_column": type_column,
                    "name_column": name_column,
                }
            )
            if type_column is None:
                continue
            rows = [
                (row.get(type_column, ""), row.get(name_column, "") if name_column else "")
                for row in reader
            ]
            lifecycle_types.update(_normalize(row[0]).upper() for row in rows)
            sources.append(_profile_source(path, profile_root, rows, "csv"))

    for path in files:
        if path.name == "ascend_task.db":
            try:
                with sqlite3.connect(path) as database:
                    columns = [
                        row[1]
                        for row in database.execute("PRAGMA table_info(AscendTask)")
                    ]
                    schemas.append(
                        {
                            "path": str(path.relative_to(profile_root)),
                            "format": "sqlite",
                            "table": "AscendTask",
                            "columns": columns,
                        }
                    )
                    if "device_task_type" not in columns:
                        continue
                    selected = ["device_task_type"]
                    if "host_task_type" in columns:
                        selected.append("host_task_type")
                    rows = []
                    for row in database.execute(
                        f"SELECT {', '.join(selected)} FROM AscendTask"
                    ):
                        device_type = row[0] or ""
                        host_type = row[1] or "" if len(row) == 2 else ""
                        rows.append((device_type, ""))
                        lifecycle_types.add(_normalize(host_type).upper())
                    sources.append(
                        _profile_source(path, profile_root, rows, "sqlite-ascend-task")
                    )
            except sqlite3.Error as error:
                schemas.append(
                    {
                        "path": str(path.relative_to(profile_root)),
                        "format": "sqlite",
                        "table": "AscendTask",
                        "error": str(error),
                    }
                )
        elif path.name == "ai_core_op_summary.db":
            try:
                with sqlite3.connect(path) as database:
                    columns = [
                        row[1]
                        for row in database.execute("PRAGMA table_info(ge_summary)")
                    ]
                    schemas.append(
                        {
                            "path": str(path.relative_to(profile_root)),
                            "format": "sqlite",
                            "table": "ge_summary",
                            "columns": columns,
                        }
                    )
                    required = {"task_type", "op_name", "op_type"}
                    if not required.issubset(columns):
                        continue
                    rows = [
                        (task_type or "", op_name or "", op_type or "")
                        for task_type, op_name, op_type in database.execute(
                            "SELECT task_type, op_name, op_type FROM ge_summary"
                        )
                    ]
                    source = _profile_source(
                        path,
                        profile_root,
                        [(task_type, name) for task_type, name, _ in rows],
                        "sqlite-ge-summary",
                    )
                    source["target_add_tasks"] = sum(
                        1
                        for task_type, name, op_type in rows
                        if _is_ai_core_task(task_type)
                        and _is_target_add(name, op_type)
                    )
                    source["op_types"] = dict(
                        sorted(Counter(op_type for _, _, op_type in rows).items())
                    )
                    sources.append(source)
            except sqlite3.Error as error:
                schemas.append(
                    {
                        "path": str(path.relative_to(profile_root)),
                        "format": "sqlite",
                        "table": "ge_summary",
                        "error": str(error),
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
    ai_core_counts = [source["ai_core_tasks"] for source in sources]
    ai_cpu_counts = [source["ai_cpu_tasks"] for source in sources]
    target_counts = [
        source["target_add_tasks"]
        for source in sources
        if source["target_add_tasks"] > 0
    ]
    ai_core_task_count = max(ai_core_counts, default=0)
    ai_cpu_task_count = max(ai_cpu_counts, default=0)
    target_add_task_count = max(target_counts, default=0)
    capture_started = "PROFILING_ENABLE" in lifecycle_types
    capture_stopped = "PROFILING_DISABLE" in lifecycle_types
    errors: list[str] = []
    if status != expected_status:
        errors.append(f"profile command status={status}, expected={expected_status}")
    if not files:
        errors.append("profile contains no files")
    if not sources:
        errors.append("profile contains no readable task evidence")
    if not capture_started or not capture_stopped:
        errors.append("profile capture lifecycle is incomplete")
    if ai_core_task_count != EXPECTED_AICORE_CALLS:
        errors.append(
            f"AICore/AIVector tasks={ai_core_task_count}, "
            f"expected {EXPECTED_AICORE_CALLS}"
        )
    if target_add_task_count != EXPECTED_AICORE_CALLS:
        errors.append(
            f"attributed target Add tasks={target_add_task_count}, "
            f"expected {EXPECTED_AICORE_CALLS}"
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
        "files": [
            str(path.relative_to(profile_root))
            for path in files[:MAX_PROFILE_FILE_RECORDS]
        ],
        "files_truncated": len(files) > MAX_PROFILE_FILE_RECORDS,
        "task_schemas": schemas,
        "task_sources": sources,
        "capture_started": capture_started,
        "capture_stopped": capture_stopped,
        "ai_cpu_task_count": ai_cpu_task_count,
        "ai_core_task_count": ai_core_task_count,
        "target_add_task_count": target_add_task_count,
        "expected_target_add_task_count": EXPECTED_AICORE_CALLS,
        "target_op": TARGET_OP,
        "controller": controller,
    }


def _output(
    owner: int,
    output_type: int,
    event_seq: int,
    cohort: int,
    row: int,
    commit_seq: int,
    state: int,
    token: int,
    remaining: int,
    credit: int,
    status: int,
    *,
    flags: int = 0,
) -> dict[str, int]:
    return {
        "owner": owner,
        "type": output_type,
        "event_seq": event_seq,
        "cohort": cohort,
        "row": row,
        "commit_seq": commit_seq,
        "state": state,
        "token": token,
        "remaining": remaining,
        "credit": credit,
        "status": status,
        "reserved": 0,
        "transaction": event_seq,
        "flags": flags,
    }


def expected_outputs(owner: int) -> list[dict[str, int]]:
    outputs = [_output(owner, 1, 1, 1, -1, 0, 0, 0, 256, 256, 0)]
    for commit in range(1, 257):
        outputs.append(
            _output(
                owner,
                2,
                1,
                1,
                0,
                commit,
                1000 + commit,
                1000 + commit,
                256 - commit,
                256 - commit,
                0,
            )
        )
        if commit == 256:
            outputs.append(outputs[-1] | {"type": 3})
    outputs.append(_output(owner, 4, 1, 1, -1, 0, 0, 0, 0, 0, 1))
    outputs.append(_output(owner, 1, 2, 2, -1, 0, 0, 0, 1024, 784, 0))
    seeds = (2000, 3000, 4000, 5000)
    for commit in range(1, 257):
        for row in range(4):
            if row == 0 and commit > 16:
                continue
            outputs.append(
                _output(
                    owner,
                    2,
                    2,
                    2,
                    row,
                    commit,
                    seeds[row] + commit,
                    seeds[row] + commit,
                    256 - commit,
                    16 - commit if row == 0 else 256 - commit,
                    0,
                )
            )
            if commit == 256:
                outputs.append(outputs[-1] | {"type": 3})
    outputs.append(_output(owner, 5, 3, 2, -1, 0, 0, 0, 1, 240, 0))
    for commit in range(17, 257):
        outputs.append(
            _output(
                owner,
                2,
                2,
                2,
                0,
                commit,
                2000 + commit,
                2000 + commit,
                256 - commit,
                256 - commit,
                0,
            )
        )
        if commit == 256:
            outputs.append(outputs[-1] | {"type": 3})
    outputs.append(_output(owner, 4, 2, 2, -1, 0, 0, 0, 0, 0, 2))
    outputs.append(
        _output(
            owner,
            6,
            4,
            2,
            -1,
            EXPECTED_AICORE_CALLS,
            EXPECTED_COMMITS,
            EXPECTED_COMMITS,
            0,
            0,
            0,
            flags=1,
        )
    )
    return outputs


def verify_start(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    outputs = [parse_fields(line) for line in lines if line.startswith("P1_OUTPUT ")]
    summaries = [
        parse_fields(line) for line in lines if line.startswith("P1_SUMMARY ")
    ]
    starts = [parse_fields(line) for line in lines if line.startswith("P1_OWNER_START ")]
    isolations = [
        parse_fields(line)
        for line in lines
        if line.startswith("P1_CREDIT_ISOLATION ")
    ]
    pre_shutdown = [
        parse_fields(line)
        for line in lines
        if line.startswith("P1_PRE_SHUTDOWN ")
    ]
    errors: list[str] = []
    if len(starts) != 1:
        errors.append(f"owner starts={len(starts)}, expected 1")
    if len(summaries) != 1:
        errors.append(f"summaries={len(summaries)}, expected 1")
        return {"path": str(path), "pass": False, "errors": errors}
    summary = summaries[0]
    expected_summary = {
        "owner_starts": 1,
        "feed_calls": 4,
        "fetch_calls": EXPECTED_OUTPUTS,
        "admission_events": 2,
        "credit_events": 1,
        "shutdown_events": 1,
        "quiescent": 2,
        "rejected": 0,
        "b1_row0_commits": 256,
        "b4_row0_commits": 256,
        "b4_row1_commits": 256,
        "b4_row2_commits": 256,
        "b4_row3_commits": 256,
        "blocked_row0_commits": 16,
        "aicore_calls": EXPECTED_AICORE_CALLS,
        "total_commits": EXPECTED_COMMITS,
        "pre_shutdown_commits": EXPECTED_COMMITS,
        "pre_shutdown_fetches": EXPECTED_PRE_SHUTDOWN_OUTPUTS,
        "compile_status": 0,
        "fetch_status": 0,
        "remove_status": 0,
        "finalize_status": 0,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            errors.append(f"{key}={summary.get(key)}, expected {expected}")
    owner = summary.get("owner")
    if len(starts) == 1 and starts[0].get("owner") != owner:
        errors.append("Owner Instance differs between start and summary")
    if summary.get("host_cpu_us", 0) <= 0 or summary.get("wall_ms", 0) <= 0:
        errors.append("Host CPU or wall duration is not positive")

    if len(isolations) != 1:
        errors.append(f"credit isolation records={len(isolations)}, expected 1")
    else:
        expected_isolation = {
            "owner": owner,
            "cohort": 2,
            "blocked_row": 0,
            "blocked_commits": 16,
            "peer1_commits": 256,
            "peer2_commits": 256,
            "peer3_commits": 256,
        }
        if isolations[0] != expected_isolation:
            errors.append(
                f"credit isolation={isolations[0]}, expected={expected_isolation}"
            )
    if len(pre_shutdown) != 1:
        errors.append(f"pre-shutdown records={len(pre_shutdown)}, expected 1")
    else:
        expected_pre_shutdown = {
            "owner": owner,
            "commits": EXPECTED_COMMITS,
            "fetch_calls": EXPECTED_PRE_SHUTDOWN_OUTPUTS,
        }
        if pre_shutdown[0] != expected_pre_shutdown:
            errors.append(
                f"pre-shutdown={pre_shutdown[0]}, expected={expected_pre_shutdown}"
            )

    if owner is not None:
        expected = expected_outputs(owner)
        if len(outputs) != len(expected):
            errors.append(f"outputs={len(outputs)}, expected {len(expected)}")
        for index, (actual, wanted) in enumerate(zip(outputs, expected)):
            if actual != wanted:
                mismatches = {
                    key: {"actual": actual.get(key), "expected": value}
                    for key, value in wanted.items()
                    if actual.get(key) != value
                }
                errors.append(f"output[{index}] mismatch={mismatches}")
                break
    return {
        "path": str(path),
        "pass": not errors,
        "errors": errors,
        "summary": summary,
        "outputs": len(outputs),
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

    if profile_log is None:
        profile_run = {
            "path": None,
            "pass": False,
            "errors": ["profile mechanism log was not provided"],
            "outputs": 0,
        }
    else:
        profile_run = verify_start(profile_log)
    controller = analyze_controller_activity(
        profile_log,
        controller_log_roots or [],
        profile_run.get("outputs", 0),
    )
    profile = analyze_profile(profile_root, profile_status, controller)
    profile_owner = profile_run.get("summary", {}).get("owner")
    profile_owner_unique = profile_owner is not None and profile_owner not in owners
    if npu_after_profile is None:
        profile_hbm = None
        profile_hbm_pass = False
    else:
        profile_hbm = parse_usage_percent(npu_after_profile)
        profile_hbm_pass = profile_hbm <= before_hbm + 2
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
    result["requested_gate"] = "mechanism" if args.mechanism_only else "full_p1"
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
