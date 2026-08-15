#!/usr/bin/env python3
"""Verify P0 logs and write the target-hardware evidence summary."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


TARGET_PROCESS_POINT = "persistent_control_p0_pp"
TARGET_FLOW_FUNC = "persistent_control_p0"
MAX_PROFILE_FILE_RECORDS = 128
MAX_CONTROLLER_EVIDENCE_LINES = 128


_OWNER_START = re.compile(r"^P0_OWNER_START pid=(\d+)\s")
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


def _parse_profile_status(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) != 2 or not row[0] or row[0] in values:
                raise ValueError(f"invalid profile status row in {path}: {row}")
            values[row[0]] = row[1]
    return values


def _relative_log_path(path: Path, roots: list[Path]) -> str:
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
    if len(host_pids) != 1:
        errors.append(f"profile Host process ids={host_pids}, expected exactly one")
        host_pid = None
    else:
        host_pid = host_pids[0]

    resolved_roots = [root.resolve(strict=True) for root in log_roots]
    files: set[Path] = set()
    if host_pid is not None:
        prefix = f"device-{host_pid}_"
        for root in resolved_roots:
            if root.is_file():
                if root.name.startswith(prefix) and root.suffix == ".log":
                    files.add(root)
                continue
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
        display_path = _relative_log_path(path, resolved_roots)
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = raw_line.rstrip("\r\n")
                executor = _UDF_EXECUTOR.search(line)
                if executor is None:
                    continue
                executor_pid = int(executor.group(1))
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
                    "executor_pid": executor_pid,
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
    for kind in records:
        if len(records[kind]) != 1:
            errors.append(f"controller {kind} records={len(records[kind])}, expected 1")

    if all(len(records[kind]) == 1 for kind in records):
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
                "controller schedule finishes="
                f"{summary['schedule_finish_times']}, expected 1"
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
            _relative_log_path(path, resolved_roots) for path in sorted(files)
        ],
        "bytes_scanned": bytes_scanned,
        "expected_outputs": expected_outputs,
        "records": records,
        "evidence_lines": evidence_lines,
        "evidence_lines_truncated": sum(len(value) for value in records.values())
        > len(evidence_lines),
    }


def analyze_profile(
    profile_root: Path,
    profile_status: Path,
    controller: dict[str, Any],
) -> dict[str, Any]:
    files = sorted(path for path in profile_root.rglob("*") if path.is_file())
    csv_task_types: Counter[str] = Counter()
    db_device_task_types: Counter[str] = Counter()
    db_host_task_types: Counter[str] = Counter()
    schemas: list[dict[str, Any]] = []
    task_evidence_available = False

    for path in files:
        if not path.name.startswith("task_time_") or path.suffix != ".csv":
            continue
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
            reader = csv.DictReader(stream)
            columns = {
                _normalize(name): name for name in (reader.fieldnames or [])
            }
            type_column = next(
                (
                    columns[candidate]
                    for candidate in ("kernel_type", "task_type", "kernel_task_type")
                    if candidate in columns
                ),
                None,
            )
            schemas.append(
                {
                    "path": str(path.relative_to(profile_root)),
                    "format": "csv",
                    "type_column": type_column,
                }
            )
            if type_column is None:
                continue
            task_evidence_available = True
            for row in reader:
                csv_task_types[row.get(type_column, "")] += 1

    for path in files:
        if path.name != "ascend_task.db":
            continue
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
                task_evidence_available = True
                selected_columns = ["device_task_type"]
                if "host_task_type" in columns:
                    selected_columns.append("host_task_type")
                for row in database.execute(
                    f"SELECT {', '.join(selected_columns)} FROM AscendTask"
                ):
                    db_device_task_types[row[0] or ""] += 1
                    if len(row) == 2:
                        db_host_task_types[row[1] or ""] += 1
        except sqlite3.Error as error:
            schemas.append(
                {
                    "path": str(path.relative_to(profile_root)),
                    "format": "sqlite",
                    "table": "AscendTask",
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
    command_pass = status == expected_status
    counters = (csv_task_types, db_device_task_types, db_host_task_types)
    ai_core_task_count = max(
        (
            sum(
                count
                for task_type, count in task_types.items()
                if _is_ai_core_task(task_type)
            )
            for task_types in counters
        ),
        default=0,
    )
    ai_cpu_task_count = max(
        (
            sum(
                count
                for task_type, count in task_types.items()
                if _is_ai_cpu_task(task_type)
            )
            for task_types in counters
        ),
        default=0,
    )
    lifecycle_types = {
        _normalize(task_type).upper()
        for task_types in (csv_task_types, db_host_task_types)
        for task_type in task_types
    }
    capture_started = "PROFILING_ENABLE" in lifecycle_types
    capture_stopped = "PROFILING_DISABLE" in lifecycle_types
    core_pass = (
        command_pass
        and bool(files)
        and task_evidence_available
        and capture_started
        and capture_stopped
        and ai_core_task_count == 0
        and controller.get("pass") is True
    )
    return {
        "pass": core_pass,
        "command_pass": command_pass,
        "profile_status": status,
        "profile_status_error": status_error,
        "file_count": len(files),
        "file_bytes": sum(path.stat().st_size for path in files),
        "files": [
            str(path.relative_to(profile_root))
            for path in files[:MAX_PROFILE_FILE_RECORDS]
        ],
        "files_truncated": len(files) > MAX_PROFILE_FILE_RECORDS,
        "task_evidence_available": task_evidence_available,
        "task_schemas": schemas,
        "task_types": {
            "csv": dict(sorted(csv_task_types.items())),
            "db_device": dict(sorted(db_device_task_types.items())),
            "db_host": dict(sorted(db_host_task_types.items())),
        },
        "capture_started": capture_started,
        "capture_stopped": capture_stopped,
        "ai_cpu_task_count": ai_cpu_task_count,
        "ai_core_task_count": ai_core_task_count,
        "aicpu_task_visibility": (
            "ordinary_task_observed"
            if ai_cpu_task_count > 0
            else "flowfunc_not_exported_as_ordinary_task"
        ),
        "controller": controller,
    }


def verify_start(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    outputs = [parse_fields(line) for line in lines if line.startswith("P0_OUTPUT ")]
    summaries = [
        parse_fields(line) for line in lines if line.startswith("P0_SUMMARY ")
    ]
    starts = [parse_fields(line) for line in lines if line.startswith("P0_OWNER_START ")]
    errors: list[str] = []
    if len(starts) != 1:
        errors.append(f"owner starts={len(starts)}, expected 1")
    if len(summaries) != 1:
        errors.append(f"summaries={len(summaries)}, expected 1")
        return {"path": str(path), "pass": False, "errors": errors}
    summary = summaries[0]
    expected_summary = {
        "owner_starts": 1,
        "feed_calls": 6,
        "admission_events": 4,
        "cancel_events": 1,
        "shutdown_events": 1,
        "quiescent": 4,
        "rejected": 0,
        "generation1_commits": 1024,
        "generation2_commits": 1024,
        "generation4_commits": 1024,
        "aicore_tasks_expected": 0,
        "compile_status": 0,
        "fetch_status": 0,
        "remove_status": 0,
        "finalize_status": 0,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            errors.append(f"{key}={summary.get(key)}, expected {expected}")
    generation_three = summary.get("generation3_commits", 0)
    if not 128 <= generation_three < 1024:
        errors.append(
            f"generation3_commits={generation_three}, expected in [128, 1024)"
        )
    if summary.get("fetch_calls") != len(outputs):
        errors.append(
            f"fetch_calls={summary.get('fetch_calls')} but parsed outputs={len(outputs)}"
        )
    owner = summary.get("owner")
    if any(output.get("owner") != owner for output in outputs):
        errors.append("output crossed Owner Instance")
    if any(output.get("type") == 7 for output in outputs):
        errors.append("Device rejected a control event")

    seeds = {1: 1000, 2: 2000, 3: 3000, 4: 4000}
    commits: dict[int, list[dict[str, int]]] = {generation: [] for generation in seeds}
    for output in outputs:
        if output.get("type") == 2 and output.get("generation") in commits:
            commits[output["generation"]].append(output)
    for generation, records in commits.items():
        sequence = [record["commit_seq"] for record in records]
        if sequence != list(range(1, len(records) + 1)):
            errors.append(f"generation {generation} commit sequence is not contiguous")
        for record in records:
            expected_state = seeds[generation] + record["commit_seq"]
            if record.get("state") != expected_state:
                errors.append(
                    f"generation {generation} state={record.get('state')} "
                    f"expected={expected_state}"
                )
                break
            if record.get("remaining") != 1024 - record["commit_seq"]:
                errors.append(f"generation {generation} remaining counter mismatch")
                break

    shutdown = [output for output in outputs if output.get("type") == 6]
    if len(shutdown) != 1 or not (shutdown[0].get("flags", 0) & 1):
        errors.append("missing unique EOS-marked shutdown output")
    cancelled = [
        output
        for output in outputs
        if output.get("type") == 4 and output.get("generation") == 3
    ]
    if len(cancelled) != 1 or cancelled[0].get("status") != 1:
        errors.append("generation 3 cancellation retirement is missing")

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
    hbm_pass = all(value <= before_hbm + 2 for value in after_hbm)
    unique_owners = len(set(owners)) == len(owners) and None not in owners
    mechanism_pass = (
        placement.get("format") == "cruise-persistent-control-p0-placement-v1"
        and placement.get("pass") is True
        and len(starts) == 3
        and all(start["pass"] for start in starts)
        and unique_owners
        and hbm_pass
    )
    return {
        "format": "cruise-persistent-control-p0-v1",
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
    result["requested_gate"] = "mechanism" if args.mechanism_only else "full_p0"
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
