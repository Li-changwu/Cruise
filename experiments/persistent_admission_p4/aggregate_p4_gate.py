#!/usr/bin/env python3
"""Aggregate the exact P4 primary and regression gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(primary_root: Path, regression_root: Path) -> dict[str, Any]:
    primary_evidence = primary_root / "evidence"
    regression_evidence = regression_root / "evidence"
    primary = load(primary_evidence / "summary.json")
    primary_gate = load(primary_evidence / "gate.json")
    regression = load(regression_evidence / "summary.json")
    regression_gate = load(regression_evidence / "gate.json")
    primary_identity = primary_evidence / "source-identity.sha256"
    regression_identity = regression_evidence / "source-identity.sha256"
    identity_equal = primary_identity.read_bytes() == regression_identity.read_bytes()
    passed = (
        primary.get("pass") is True
        and primary_gate.get("pass") is True
        and regression.get("pass") is True
        and regression_gate.get("pass") is True
        and identity_equal
    )
    return {
        "gate": "P4-GATE-candidate-hardware-validated",
        "pass": passed,
        "primary": {
            "run": primary_root.name,
            "summary_sha256": sha256(primary_evidence / "summary.json"),
            "gate_sha256": sha256(primary_evidence / "gate.json"),
            "requests": primary.get("requests"),
            "feed_calls": primary.get("feed_calls"),
            "aicore_calls": primary.get("aicore_calls"),
            "total_commits": primary.get("total_commits"),
            "duration_ns": primary.get("duration_ns"),
            "device_decode_coverage_percent": primary_gate.get(
                "device_decode_coverage_percent"
            ),
        },
        "regression": {
            "run": regression_root.name,
            "summary_sha256": sha256(regression_evidence / "summary.json"),
            "gate_sha256": sha256(regression_evidence / "gate.json"),
            "requests": regression.get("requests"),
            "feed_calls": regression.get("feed_calls"),
            "aicore_calls": regression.get("aicore_calls"),
            "total_commits": regression.get("total_commits"),
            "rejected": regression.get("rejected"),
            "credit_acks": regression.get("credit_acks"),
        },
        "identity": {
            "source_identity_equal": identity_equal,
            "source_identity_sha256": sha256(primary_identity),
            "oracle_identity_record_sha256": sha256(
                primary_evidence / "oracle-identity.sha256"
            ),
        },
        "release": {
            "mutable_external_weight_views_retired": not (
                (primary_root / ".ge-external-view").exists()
                or (regression_root / ".ge-external-view").exists()
            ),
        },
        "claim_boundary": (
            "P4 Candidate Hardware Validated for serial admission and service "
            "semantics only. P5 performance and P6 concurrency remain open."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--regression-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.primary_root, args.regression_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 20


if __name__ == "__main__":
    raise SystemExit(main())
