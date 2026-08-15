#!/usr/bin/env python3
"""Verify exact Graph and GraphPp execution of the minimal custom AICore op."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


LAUNCH_PATTERN = re.compile(
    r"LaunchKernel: kernel info.*kernel_name=te_bf16materialize_", re.IGNORECASE
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_mode(mode: str, result_path: Path, output: Path, log: Path, input_sha: str) -> dict:
    if not result_path.is_file() or not log.is_file():
        return {
            "pass": False,
            "summary_pass": False,
            "output_sha256": None,
            "output_matches_input": False,
            "custom_kernel_launch_count": 0,
            "missing_files": [
                str(path) for path in (result_path, log) if not path.is_file()
            ],
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    log_text = log.read_text(encoding="utf-8", errors="replace")
    output_sha = sha256(output) if output.is_file() else None
    launch_count = len(LAUNCH_PATTERN.findall(log_text))
    passed = (
        result.get("pass") is True
        and result.get("mode") == mode
        and result.get("output_exact") is True
        and output_sha == input_sha
        and launch_count >= 1
    )
    return {
        "pass": passed,
        "summary_pass": result.get("pass") is True,
        "output_sha256": output_sha,
        "output_matches_input": output_sha == input_sha,
        "custom_kernel_launch_count": launch_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--graph-result", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--graph-log", type=Path, required=True)
    parser.add_argument("--dataflow-result", type=Path, required=True)
    parser.add_argument("--dataflow-output", type=Path, required=True)
    parser.add_argument("--dataflow-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_sha = sha256(args.input)
    modes = {
        "graph": check_mode(
            "graph", args.graph_result, args.graph_output, args.graph_log, input_sha
        ),
        "dataflow": check_mode(
            "dataflow",
            args.dataflow_result,
            args.dataflow_output,
            args.dataflow_log,
            input_sha,
        ),
    }
    result = {
        "gate": "P5-CUSTOM-AICORE-GRAPHPP",
        "pass": all(value["pass"] for value in modes.values()),
        "input_sha256": input_sha,
        "modes": modes,
        "claim_boundary": (
            "Single BF16 identity custom AICore op only; no FIA, Owner, "
            "or P5 performance claim."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("P5_CUSTOM_AICORE_GRAPHPP " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
