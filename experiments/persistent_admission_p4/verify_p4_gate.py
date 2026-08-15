#!/usr/bin/env python3
"""Verify P4 service invariants and exact primary recurrence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXACT_FIELDS = (
    "commits",
    "retired",
    "cancelled",
    "final_position",
    "final_page",
    "final_checksum",
    "finish_reason",
    "tokens",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return value


def states(summary: dict[str, Any]) -> list[dict[str, Any]]:
    value = summary.get("request_states")
    if not isinstance(value, list) or not value:
        raise ValueError("summary has no request_states")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("request_states must contain JSON objects")
    return value


def mismatch(items: list[dict[str, Any]], field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        items.append({"field": field, "actual": actual, "expected": expected})


def verify_primary(
    summary: dict[str, Any], oracle: dict[str, Any], mismatches: list[dict[str, Any]]
) -> None:
    service_states = states(summary)
    oracle_states = states(oracle)
    if len(oracle_states) != 1 or oracle.get("pass") is not True:
        mismatches.append({"field": "oracle", "actual": "not one passing request"})
        return
    reference = oracle_states[0]
    mismatch(mismatches, "requests", len(service_states), 32)
    mismatch(mismatches, "feed_calls", summary.get("feed_calls"), 33)
    mismatch(mismatches, "aicore_calls", summary.get("aicore_calls"), 3064)
    mismatch(mismatches, "total_commits", summary.get("total_commits"), 8192)
    mismatch(mismatches, "total_retired", summary.get("total_retired"), 32)
    mismatch(mismatches, "single_token_outputs", summary.get("single_token_outputs"), 8192)
    mismatch(mismatches, "max_pending_requests", summary.get("max_pending_requests"), 4)
    mismatch(mismatches, "max_active_rows", summary.get("max_active_rows"), 4)
    mismatch(mismatches, "rejected", summary.get("rejected"), 0)
    mismatch(mismatches, "oracle.model_calls", oracle.get("model_calls"), 383)
    row_counts = Counter(int(item.get("row", -1)) for item in service_states)
    mismatch(mismatches, "row_reuse_counts", dict(sorted(row_counts.items())), {0: 8, 1: 8, 2: 8, 3: 8})
    for index, state in enumerate(service_states):
        for field in EXACT_FIELDS:
            actual = state.get(field)
            expected = reference.get(field)
            if actual != expected:
                mismatches.append(
                    {
                        "request_index": index,
                        "field": field,
                        "actual": actual,
                        "expected": expected,
                    }
                )
        mismatch(
            mismatches,
            f"request_states[{index}].stream_messages",
            state.get("stream_messages"),
            256,
        )
        timestamps = state.get("token_timestamps_ns")
        if not isinstance(timestamps, list) or len(timestamps) != 256:
            mismatches.append(
                {
                    "request_index": index,
                    "field": "token_timestamps_ns",
                    "actual": None if not isinstance(timestamps, list) else len(timestamps),
                    "expected": 256,
                }
            )
        elif any(right < left for left, right in zip(timestamps, timestamps[1:])):
            mismatches.append(
                {"request_index": index, "field": "token_timestamps_ns.order"}
            )


def verify_regression(summary: dict[str, Any], mismatches: list[dict[str, Any]]) -> None:
    request_states = states(summary)
    mismatch(mismatches, "requests", len(request_states), 26)
    mismatch(mismatches, "feed_calls", summary.get("feed_calls"), 36)
    aicore_calls = summary.get("aicore_calls")
    if not isinstance(aicore_calls, int) or not 110 <= aicore_calls <= 131:
        mismatches.append(
            {
                "field": "aicore_calls",
                "actual": aicore_calls,
                "expected": "integer in [110, 131]",
            }
        )
    mismatch(mismatches, "total_commits", summary.get("total_commits"), 194)
    mismatch(mismatches, "total_retired", summary.get("total_retired"), 26)
    mismatch(mismatches, "single_token_outputs", summary.get("single_token_outputs"), 194)
    mismatch(mismatches, "max_active_rows", summary.get("max_active_rows"), 4)
    mismatch(mismatches, "max_pending_requests", summary.get("max_pending_requests"), 8)
    mismatch(mismatches, "rejected", summary.get("rejected"), 4)
    mismatch(mismatches, "credit_acks", summary.get("credit_acks"), 4)
    mismatch(mismatches, "rejected_statuses", summary.get("rejected_statuses"), {"3": 4})
    tag_counts = Counter(str(item.get("tag")) for item in request_states)
    mismatch(
        mismatches,
        "tag_counts",
        dict(sorted(tag_counts.items())),
        {"burst": 8, "cancel": 1, "eos": 1, "overload": 8, "short": 8},
    )
    eos = [item for item in request_states if item.get("tag") == "eos"]
    if len(eos) != 1 or eos[0].get("tokens") != [151645] or eos[0].get("finish_reason") != 1:
        mismatches.append({"field": "natural_eos", "actual": eos})
    cancelled = [item for item in request_states if item.get("tag") == "cancel"]
    if len(cancelled) != 1 or cancelled[0].get("commits") != 1 or cancelled[0].get("finish_reason") != 3:
        mismatches.append({"field": "cancel_after_commit", "actual": cancelled})
    overload = [item for item in request_states if item.get("tag") == "overload"]
    if sorted(int(item.get("rejected_admissions", -1)) for item in overload) != [0, 0, 0, 0, 1, 1, 1, 1]:
        mismatches.append({"field": "overload_retry", "actual": overload})


def verify(summary: dict[str, Any], oracle: dict[str, Any] | None = None) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    mismatch(mismatches, "summary.pass", summary.get("pass"), True)
    mismatch(mismatches, "gate", summary.get("gate"), "P4-SERVICE")
    mismatch(mismatches, "protocol_errors", summary.get("protocol_errors"), 0)
    mismatch(mismatches, "shutdown", summary.get("shutdown"), True)
    scenario = summary.get("scenario")
    if scenario == "primary-c4":
        if oracle is None:
            mismatches.append({"field": "oracle", "actual": "missing"})
        else:
            verify_primary(summary, oracle, mismatches)
    elif scenario == "regression":
        verify_regression(summary, mismatches)
    else:
        mismatches.append({"field": "scenario", "actual": scenario})
    return {
        "gate": "P4-GATE-quantum-boundary-serial-admission",
        "pass": not mismatches,
        "scenario": scenario,
        "device_decode_coverage_percent": 100 if not mismatches else 0,
        "mismatches": mismatches,
        "claim_boundary": (
            "P4 serial admission and service semantics only; no P5 performance "
            "qualification or P6 concurrent Prefill/Decode claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(load(args.summary), load(args.oracle) if args.oracle else None)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 20


if __name__ == "__main__":
    raise SystemExit(main())
