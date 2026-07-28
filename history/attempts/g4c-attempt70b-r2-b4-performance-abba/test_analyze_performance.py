#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


FIELDS = (
    "phase",
    "k",
    "iteration",
    "order",
    "position",
    "route",
    "wall_us",
    "cpu_us",
    "host_model_submissions",
    "feed_calls",
    "fetch_calls",
)


def make_rows() -> list[dict[str, str | int]]:
    rows = []
    for k in (2, 4, 8):
        for phase, count in (("warmup", 3), ("measure", 15)):
            for iteration in range(count):
                order = "HD" if iteration % 2 == 0 else "DH"
                routes = ("host", "device") if order == "HD" else ("device", "host")
                for position, route in enumerate(routes):
                    host = route == "host"
                    rows.append(
                        {
                            "phase": phase,
                            "k": k,
                            "iteration": iteration,
                            "order": order,
                            "position": position,
                            "route": route,
                            "wall_us": k * (10000 if host else 5000) + iteration,
                            "cpu_us": k * (2000 if host else 1000) + iteration,
                            "host_model_submissions": k if host else 1,
                            "feed_calls": 0 if host else 1,
                            "fetch_calls": 0 if host else 1,
                        }
                    )
    return rows


def run_case(root: Path, name: str, rows: list[dict[str, str | int]]) -> dict:
    input_path = root / f"{name}.tsv"
    output_path = root / f"{name}.json"
    with input_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("analyze_performance.py")),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    result = json.loads(output_path.read_text(encoding="ascii"))
    result["exit_status"] = completed.returncode
    return result


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="attempt70b-analyzer-") as tmp:
        root = Path(tmp)
        passing = make_rows()
        pass_result = run_case(root, "pass", passing)
        if pass_result["exit_status"] != 0 or not pass_result["pass"]:
            return 10

        bad_measured = make_rows()
        next(
            row
            for row in bad_measured
            if row["phase"] == "measure" and row["route"] == "device"
        )["feed_calls"] = 0
        measured_result = run_case(root, "bad-measured", bad_measured)
        if measured_result["exit_status"] != 1 or measured_result["pass"]:
            return 11

        bad_warmup = make_rows()
        next(
            row
            for row in bad_warmup
            if row["phase"] == "warmup" and row["route"] == "device"
        )["route"] = "host"
        warmup_result = run_case(root, "bad-warmup", bad_warmup)
        if warmup_result["exit_status"] != 1 or warmup_result["pass"]:
            return 12

    print("attempt70b-analyzer-selftest\tPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
