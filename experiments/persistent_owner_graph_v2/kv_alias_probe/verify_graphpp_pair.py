#!/usr/bin/env python3
"""Verify exact ordinary Graph and public GraphPp KV alias execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_mode(mode: str, result_path: Path, output: Path, expected_sha: str) -> dict:
    if not result_path.is_file() or not output.is_file():
        return {
            "pass": False,
            "missing_files": [
                str(path) for path in (result_path, output) if not path.is_file()
            ],
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output_sha = sha256(output)
    passed = (
        result.get("pass") is True
        and result.get("mode") == mode
        and result.get("model_load_status") == 0
        and result.get("model_valid") is True
        and result.get("output_count") == 2
        and result.get("key_exact") is True
        and result.get("value_exact") is True
        and result.get("output_exact") is True
        and output_sha == expected_sha
    )
    return {
        "pass": passed,
        "summary_pass": result.get("pass") is True,
        "output_sha256": output_sha,
        "output_matches_oracle": output_sha == expected_sha,
        "elapsed_ms": result.get("elapsed_ms"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-output", type=Path, required=True)
    parser.add_argument("--graph-result", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--dataflow-result", type=Path, required=True)
    parser.add_argument("--dataflow-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_sha = sha256(args.expected_output)
    modes = {
        "graph": check_mode(
            "graph", args.graph_result, args.graph_output, expected_sha
        ),
        "dataflow": check_mode(
            "dataflow", args.dataflow_result, args.dataflow_output, expected_sha
        ),
    }
    result = {
        "gate": "V2-KV-ALIAS-GRAPHPP-PAIR",
        "pass": all(value["pass"] for value in modes.values()),
        "expected_output_sha256": expected_sha,
        "ordinary_graph_matches_graphpp": (
            modes["graph"].get("output_sha256")
            == modes["dataflow"].get("output_sha256")
            == expected_sha
        ),
        "modes": modes,
        "claim_boundary": (
            "Single-layer B4/K384 KV update execution only; no full-model, "
            "shared-lease, Device-copy, or P5 performance claim."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("V2_KV_ALIAS_GRAPHPP " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
