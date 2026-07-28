#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


BYTE_HEADERS = {
    "bytes",
    "size(byte)",
    "size(bytes)",
    "data size",
    "data size(byte)",
    "data size(bytes)",
    "datasize",
    "data_size",
    "trans size",
    "transfer size",
}
COPY_TOKENS = ("memcpy", "memory copy", "memorycopy", "h2d", "d2h")


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _integer(value: str) -> int | None:
    cleaned = value.strip().replace(",", "")
    match = re.fullmatch(r"([0-9]+)(?:\.0+)?", cleaned)
    return int(match.group(1)) if match else None


def _direction(text: str) -> str | None:
    lowered = text.lower().replace("_", " ").replace("-", " ")
    if "h2d" in lowered or "host to device" in lowered:
        return "host_to_device"
    if "d2h" in lowered or "device to host" in lowered:
        return "device_to_host"
    return None


def scan_route(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    headers: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    totals = {"host_to_device": 0, "device_to_host": 0}
    counts = {"host_to_device": 0, "device_to_host": 0}
    for path in files:
        try:
            stream = path.open("r", encoding="utf-8-sig", newline="", errors="replace")
        except OSError:
            continue
        with stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames or []
            normalized = {_normalize(name): name for name in fieldnames}
            headers.append(
                {
                    "path": str(path),
                    "columns": fieldnames,
                }
            )
            byte_columns = [
                original
                for name, original in normalized.items()
                if name in BYTE_HEADERS
            ]
            if not byte_columns:
                continue
            for row_number, row in enumerate(reader, start=2):
                row_text = " ".join(str(value or "") for value in row.values())
                lowered = row_text.lower()
                if not any(token in lowered for token in COPY_TOKENS):
                    continue
                direction = _direction(row_text)
                byte_values = [
                    (column, _integer(str(row.get(column, ""))))
                    for column in byte_columns
                ]
                byte_values = [item for item in byte_values if item[1] is not None]
                candidates.append(
                    {
                        "path": str(path),
                        "row": row_number,
                        "direction": direction or "unclassified",
                        "byte_values": [
                            {"column": column, "value": value}
                            for column, value in byte_values
                        ],
                        "operation": row_text[:512],
                    }
                )
                if direction is None or len(byte_values) != 1:
                    continue
                value = byte_values[0][1]
                assert value is not None
                totals[direction] += value
                counts[direction] += 1

    both_directions = all(counts[direction] > 0 for direction in counts)
    status = "observed" if both_directions else "not_observed"
    reason = None
    if status == "not_observed":
        if not files:
            reason = "msprof export contains no CSV reports"
        elif not candidates:
            reason = "no exported CSV row exposes both a memcpy operation and byte field"
        else:
            reason = (
                "exported memcpy rows do not expose at least one unambiguous "
                "byte value for both H2D and D2H directions"
            )
    return {
        "status": status,
        "reason": reason,
        "root": str(root),
        "csv_files_examined": len(files),
        "headers": headers,
        "directional_row_counts": counts,
        "host_to_device_bytes": totals["host_to_device"] if status == "observed" else None,
        "device_to_host_bytes": totals["device_to_host"] if status == "observed" else None,
        "total_bytes": sum(totals.values()) if status == "observed" else None,
        "candidate_rows": candidates[:100],
        "candidate_rows_truncated": len(candidates) > 100,
    }


def summarize(old_root: Path, new_root: Path) -> dict[str, Any]:
    routes = {
        "old": scan_route(old_root),
        "new": scan_route(new_root),
    }
    observed = all(route["status"] == "observed" for route in routes.values())
    if observed:
        status = "observed"
        reason = None
    else:
        status = "not_observed"
        reason = "; ".join(
            f"{name}: {route['reason']}"
            for name, route in routes.items()
            if route["status"] != "observed"
        )
    return {
        "schema_version": 1,
        "metric": "profiler-observed directional Host-Device memcpy bytes for the full profiled process",
        "status": status,
        "reason": reason,
        "routes": routes,
        "logical_abi_bytes_used_as_transfer": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize(args.old_root, args.new_root)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"MSPROF_TRANSFER_SUMMARY_INVALID: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
