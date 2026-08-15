#!/usr/bin/env python3
"""Compare one Persistent Owner run with an independent cold Graph oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STATE_FIELDS = (
    "row",
    "commits",
    "retired",
    "cancelled",
    "final_position",
    "final_page",
    "final_checksum",
    "finish_reason",
    "tokens",
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return value


def request_map(summary: dict) -> dict[tuple[int, int], dict]:
    states = summary.get("request_states")
    if not isinstance(states, list) or not states:
        raise ValueError("summary has no request states")
    mapped: dict[tuple[int, int], dict] = {}
    for state in states:
        if not isinstance(state, dict):
            raise ValueError("request state must be a JSON object")
        key = (int(state["request"]), int(state["generation"]))
        if key in mapped:
            raise ValueError(f"duplicate request state: {key}")
        mapped[key] = state
    return mapped


def compare(owner: dict, oracle: dict) -> dict:
    mismatches: list[dict[str, object]] = []
    if owner.get("pass") is not True:
        mismatches.append({"field": "owner.pass", "actual": owner.get("pass")})
    if oracle.get("pass") is not True:
        mismatches.append({"field": "oracle.pass", "actual": oracle.get("pass")})
    scenario = owner.get("scenario")
    if scenario != oracle.get("scenario"):
        mismatches.append(
            {
                "field": "scenario",
                "owner": scenario,
                "oracle": oracle.get("scenario"),
            }
        )
    owner_states = request_map(owner)
    oracle_states = request_map(oracle)
    if owner_states.keys() != oracle_states.keys():
        mismatches.append(
            {
                "field": "request_keys",
                "owner": sorted(owner_states),
                "oracle": sorted(oracle_states),
            }
        )
    for key in sorted(owner_states.keys() & oracle_states.keys()):
        for field in STATE_FIELDS:
            actual = owner_states[key].get(field)
            expected = oracle_states[key].get(field)
            if actual != expected:
                mismatches.append(
                    {
                        "request": list(key),
                        "field": field,
                        "owner": actual,
                        "oracle": expected,
                    }
                )
    expected_feeds = len(owner_states) + 1
    if owner.get("feed_calls") != expected_feeds:
        mismatches.append(
            {
                "field": "owner.feed_calls",
                "owner": owner.get("feed_calls"),
                "expected": expected_feeds,
            }
        )
    if owner.get("aicore_calls") != oracle.get("model_calls"):
        mismatches.append(
            {
                "field": "model_calls",
                "owner": owner.get("aicore_calls"),
                "oracle": oracle.get("model_calls"),
            }
        )
    return {
        "gate": "P3-GATE-owner-vs-cold-graph",
        "pass": not mismatches,
        "scenario": scenario,
        "request_count": len(owner_states),
        "compared_fields": list(STATE_FIELDS),
        "device_decode_coverage_percent": 100 if not mismatches else 0,
        "mismatches": mismatches,
        "claim_boundary": (
            "Exact Owner versus independent cold Graph recurrence only; "
            "performance and three-start qualification remain separate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(load(args.owner), load(args.oracle))
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 20


if __name__ == "__main__":
    raise SystemExit(main())
