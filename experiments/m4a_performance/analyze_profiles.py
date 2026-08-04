#!/usr/bin/env python3
"""Reduce transient M4a msprof exports to bounded attribution evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any


ROUTES = ("eager", "graph", "cruise")
MAX_TIMELINE_ROWS = 250_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "").strip())
    except ValueError:
        return None


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "max": max(values),
    }


def _column(columns: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


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


def _merged_idle_gaps(intervals: list[tuple[float, float]]) -> list[float]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged_end = ordered[0][1]
    gaps: list[float] = []
    for start, end in ordered[1:]:
        if start > merged_end:
            gaps.append(start - merged_end)
        merged_end = max(merged_end, end)
    return gaps


def _analyze_route(root: Path, route: str, evidence: Path) -> dict[str, Any]:
    route_root = root / route
    files = sorted(path for path in route_root.rglob("*") if path.is_file())
    inventory = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    task_files = [path for path in files if path.name.startswith("task_time_")]
    task_databases = [path for path in files if path.name == "ascend_task.db"]
    task_types: Counter[str] = Counter()
    intervals: list[tuple[float, float]] = []
    timeline_rows: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []

    def record_task(
        task_type: str,
        start_value: str | int | float | None,
        duration_value: str | int | float | None,
        name: str,
        *,
        time_scale: float = 1.0,
    ) -> None:
        task_types[task_type] += 1
        if not _is_ai_core_task(task_type):
            return
        start = _number(str(start_value)) if start_value is not None else None
        duration = (
            _number(str(duration_value)) if duration_value is not None else None
        )
        if start is None or duration is None or duration < 0:
            return
        start *= time_scale
        duration *= time_scale
        intervals.append((start, start + duration))
        if len(timeline_rows) < MAX_TIMELINE_ROWS:
            timeline_rows.append(
                {
                    "start_us": start,
                    "duration_us": duration,
                    "task_type": task_type,
                    "name": name,
                }
            )

    for path in task_files:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames or []
            columns = {_normalize(name): name for name in fieldnames}
            type_column = _column(
                columns, ("kernel_type", "task_type", "kernel_task_type")
            )
            start_column = _column(
                columns,
                (
                    "task_start_time_us",
                    "start_time_us",
                    "task_start_time",
                    "start_time",
                ),
            )
            duration_column = _column(
                columns,
                (
                    "task_duration_us",
                    "duration_us",
                    "task_duration",
                    "duration",
                ),
            )
            name_column = _column(
                columns, ("op_name", "kernel_name", "task_name", "name")
            )
            schemas.append(
                {
                    "path": str(path.relative_to(root)),
                    "columns": fieldnames,
                    "type_column": type_column,
                    "start_column": start_column,
                    "duration_column": duration_column,
                    "name_column": name_column,
                }
            )
            if type_column is None or start_column is None or duration_column is None:
                continue
            for row in reader:
                task_type = row.get(type_column, "")
                record_task(
                    task_type,
                    row.get(start_column),
                    row.get(duration_column),
                    row.get(name_column, "") if name_column else "",
                )

    for path in task_databases:
        try:
            with sqlite3.connect(path) as database:
                columns = [
                    row[1]
                    for row in database.execute("PRAGMA table_info(AscendTask)")
                ]
                required = {
                    "device_task_type",
                    "start_time",
                    "duration",
                    "host_task_type",
                }
                schemas.append(
                    {
                        "path": str(path.relative_to(root)),
                        "format": "sqlite",
                        "table": "AscendTask",
                        "columns": columns,
                    }
                )
                if not required.issubset(columns):
                    continue
                rows = database.execute(
                    "SELECT device_task_type, start_time, duration, host_task_type "
                    "FROM AscendTask"
                )
                for task_type, start, duration, name in rows:
                    record_task(
                        task_type or "",
                        start,
                        duration,
                        name or "",
                        time_scale=0.001,
                    )
        except sqlite3.Error as error:
            schemas.append(
                {
                    "path": str(path.relative_to(root)),
                    "format": "sqlite",
                    "table": "AscendTask",
                    "error": str(error),
                }
            )

    timeline_path = evidence / f"profile-{route}-ai-core-timeline.csv"
    if timeline_rows:
        with timeline_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("start_us", "duration_us", "task_type", "name"),
            )
            writer.writeheader()
            writer.writerows(timeline_rows)

    gaps = _merged_idle_gaps(intervals)
    return {
        "route": route,
        "raw_profile_files": inventory,
        "task_time_schemas": schemas,
        "task_types": dict(sorted(task_types.items())),
        "ai_core_tasks_observed": bool(intervals),
        "ai_core_task_count": len(intervals),
        "ai_core_task_duration_us": _summary(
            [end - start for start, end in intervals]
        ),
        "ai_core_idle_gap_us": _summary(gaps),
        "timeline_rows_retained": len(timeline_rows),
        "timeline_truncated": len(intervals) > len(timeline_rows),
        "timeline_path": str(timeline_path) if timeline_rows else None,
        "ai_core_gap_status": "observed" if intervals else "not_observed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=ROUTES)
    args = parser.parse_args()

    profile_root = args.profile_root.resolve(strict=True)
    evidence = args.evidence_dir.resolve(strict=True)
    required_routes = tuple(args.routes)
    routes = {
        route: _analyze_route(profile_root, route, evidence)
        for route in required_routes
    }
    with args.status.open(newline="", encoding="utf-8") as stream:
        runner_status = {
            row["case"]: int(row["exit_status"])
            for row in csv.DictReader(stream, delimiter="\t")
            if row.get("case", "").startswith("profile-")
        }
    result = {
        "schema_version": 1,
        "gate": "M4a representative dynamic msprof attribution",
        "required_routes": list(required_routes),
        "routes": routes,
        "runner_status": runner_status,
        "comparison_status": (
            "observed_all_routes"
            if all(record["ai_core_tasks_observed"] for record in routes.values())
            else "not_observed_by_current_msprof_path"
        ),
        "claim_boundary": (
            "Idle gaps are reported only when exported CANN task records contain "
            "timestamped AI Core tasks. Host logical timing is never substituted."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    profile_statuses_ok = all(
        runner_status.get(f"profile-{route}-benchmark") == 0
        and runner_status.get(f"profile-{route}-dynamic-env") == 0
        and runner_status.get(f"profile-{route}-dynamic-socket") == 0
        and runner_status.get(f"profile-{route}-workload") == 0
        and runner_status.get(f"profile-{route}-start") == 0
        and runner_status.get(f"profile-{route}-stop") == 0
        and runner_status.get(f"profile-{route}-quit") == 0
        and runner_status.get(f"profile-{route}-msprof") == 0
        and runner_status.get(f"profile-{route}-export") == 0
        for route in required_routes
    )
    return int(not (profile_statuses_ok and result["comparison_status"] == "observed_all_routes"))


if __name__ == "__main__":
    raise SystemExit(main())
