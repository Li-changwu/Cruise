#!/usr/bin/env python3
"""Fail fast after the first formal Graph/Owner P5 pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.persistent_decode_p5.verify_p5_gate import _semantics, _sha256


def verify_pair(graph_path: Path, owner_path: Path) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    owner_metrics = owner.get("owner_metrics") or {}
    owner_delta = owner.get("owner_counter_delta") or {}
    checks = {
        "formal_pair_labels": graph.get("run_label") == "graph-1"
        and owner.get("run_label") == "owner-1",
        "routes_exact": graph.get("route") == "graph"
        and owner.get("route") == "owner",
        "both_starts_passed": graph.get("pass") is True
        and owner.get("pass") is True,
        "semantics_exact": _semantics(graph) == _semantics(owner),
        "persistent_owner_identity": owner_metrics.get("route")
        == "persistent_device_model_owner"
        and owner_metrics.get("host_visible_decode_epoch") is False
        and owner_metrics.get("host_token_step_api") is False
        and owner_metrics.get("forbidden_runtime_modules") == [],
        "owner_counter_coverage": owner_delta.get("admission_events") == 32
        and owner_delta.get("aicore_calls") == 3064
        and owner_delta.get("commit_events") == 8192
        and owner_delta.get("host_decode_steps") == 0,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "gate": "P5-first-pair-smoke",
        "input_sha256": {
            str(graph_path): _sha256(graph_path),
            str(owner_path): _sha256(owner_path),
        },
        "checks": checks,
        "pass": passed,
        "decision": "continue-six-start-matrix" if passed else "stop-before-owner-2",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_pair(
        args.graph.resolve(strict=True), args.owner.resolve(strict=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
