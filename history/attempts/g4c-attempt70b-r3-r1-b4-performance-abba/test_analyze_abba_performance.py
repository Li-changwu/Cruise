#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from analyze_abba_performance import BLOCKS


FIELDS = (
    "block",
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


def write_blocks(root: Path) -> dict[str, Path]:
    paths = {}
    for ordinal, (name, spec) in enumerate(BLOCKS.items()):
        path = root / f"{name}.tsv"
        paths[name] = path
        with path.open("w", newline="", encoding="ascii") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            for k in (2, 4, 8):
                for phase, count in (("warmup", 3), ("measure", spec["repeats"])):
                    offset = 0 if phase == "warmup" or spec["block"] == 1 else 8
                    for sample in range(count):
                        host = spec["route"] == "host"
                        writer.writerow(
                            {
                                "block": spec["block"],
                                "phase": phase,
                                "k": k,
                                "iteration": offset + sample,
                                "order": spec["order"],
                                "position": spec["position"],
                                "route": spec["route"],
                                "wall_us": k * (10000 if host else 5000) + sample,
                                "cpu_us": k * (2000 if host else 1000) + sample,
                                "host_model_submissions": k if host else 1,
                                "feed_calls": 0 if host else 1,
                                "fetch_calls": 0 if host else 1,
                            }
                        )
        os.utime(path, (1000 + ordinal, 1000 + ordinal))
    return paths


def run_case(root: Path, name: str, paths: dict[str, Path]) -> dict:
    output = root / f"{name}.json"
    command = [sys.executable, str(Path(__file__).with_name("analyze_abba_performance.py"))]
    for block, path in paths.items():
        command.extend((f"--{block}", str(path)))
    command.extend(("--output", str(output)))
    completed = subprocess.run(command, check=False, stdout=subprocess.DEVNULL)
    result = json.loads(output.read_text(encoding="ascii"))
    result["exit_status"] = completed.returncode
    return result


def mutate(path: Path, predicate, field: str, value: str) -> None:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    target = next(row for row in rows if predicate(row))
    target[field] = value
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="attempt70b-r3-r1-analyzer-") as tmp:
        root = Path(tmp)
        paths = write_blocks(root)
        passing = run_case(root, "pass", paths)
        if passing["exit_status"] != 0 or not passing["pass"]:
            return 10

        paths = write_blocks(root)
        mutate(paths["d1"], lambda row: row["phase"] == "measure", "feed_calls", "0")
        bad_semantics = run_case(root, "bad-semantics", paths)
        if bad_semantics["exit_status"] != 1 or bad_semantics["pass"]:
            return 11

        paths = write_blocks(root)
        mutate(paths["h2"], lambda row: True, "order", "HD")
        bad_order = run_case(root, "bad-order", paths)
        if bad_order["exit_status"] != 1 or bad_order["pass"]:
            return 12

    print("attempt70b-r3-r1-abba-analyzer-selftest\tPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
