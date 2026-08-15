#!/usr/bin/env python3
"""Independently verify the six-start P5 Graph versus Owner matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.persistent_decode_p5.run_p5_benchmark import (
    START_ORDER,
    load_workload,
    summarize,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantics(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request_index": item.get("request_index"),
            "tokens": item.get("tokens"),
            "text": item.get("text"),
            "finish_reason": item.get("finish_reason"),
            "done": item.get("done"),
            "usage": {
                key: (item.get("usage") or {}).get(key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "stream_chunk_sizes": item.get("stream_chunk_sizes"),
        }
        for item in result.get("primary", [])
    ]


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for result in results for record in result["primary"]]
    output_tokens = sum(len(record["tokens"]) for record in records)
    duration_seconds = sum(float(result["metrics"]["duration_ms"]) for result in results) / 1000
    cpu_seconds = sum(
        float(result["process_tree_cpu"]["cpu_seconds"]) for result in results
    )
    return {
        "independent_starts": len(results),
        "request_count": len(records),
        "output_tokens": output_tokens,
        "tpot_ms": summarize(record["tpot_ms"] for record in records),
        "ttft_ms": summarize(record["ttft_ms"] for record in records),
        "output_tokens_per_second": output_tokens / duration_seconds,
        "host_cpu_seconds": cpu_seconds,
        "host_cpu_ms_per_output_token": cpu_seconds * 1000 / output_tokens,
        "per_start_output_tokens_per_second": summarize(
            result["metrics"]["output_tokens_per_second"] for result in results
        ),
        "per_start_host_cpu_ms_per_output_token": summarize(
            result["metrics"]["host_cpu_ms_per_output_token"] for result in results
        ),
    }


def _reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline metric must be positive")
    return (baseline - candidate) / baseline * 100


def _increase(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline metric must be positive")
    return (candidate - baseline) / baseline * 100


def verify(paths: list[Path], workload_path: Path) -> dict[str, Any]:
    workload = load_workload(workload_path)
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    labels = [result.get("run_label") for result in loaded]
    grouped = {
        route: [result for result in loaded if result.get("route") == route]
        for route in ("graph", "owner")
    }
    canonical = _semantics(loaded[0]) if loaded else []
    semantic_mismatches = {
        str(paths[index]): _semantics(result)
        for index, result in enumerate(loaded)
        if _semantics(result) != canonical
    }
    aggregates = {
        route: _aggregate(results)
        for route, results in grouped.items()
        if len(results) == 3 and all(result.get("primary") for result in results)
    }
    threshold_values: dict[str, float] = {}
    if set(aggregates) == {"graph", "owner"}:
        graph = aggregates["graph"]
        owner = aggregates["owner"]
        threshold_values = {
            "host_cpu_per_token_reduction_percent": _reduction(
                float(graph["host_cpu_ms_per_output_token"]),
                float(owner["host_cpu_ms_per_output_token"]),
            ),
            "tpot_p50_improvement_percent": _reduction(
                float(graph["tpot_ms"]["p50"]), float(owner["tpot_ms"]["p50"])
            ),
            "tpot_p95_improvement_percent": _reduction(
                float(graph["tpot_ms"]["p95"]), float(owner["tpot_ms"]["p95"])
            ),
            "output_throughput_improvement_percent": _increase(
                float(graph["output_tokens_per_second"]),
                float(owner["output_tokens_per_second"]),
            ),
        }
    threshold_checks = {
        key: threshold_values.get(key, float("-inf")) >= expected
        for key, expected in workload.thresholds.items()
    }
    owner_identity = [result.get("owner_metrics") for result in grouped["owner"]]
    execution_checks = {
        "interleaved_order_exact": labels == list(START_ORDER),
        "three_independent_starts_per_route": all(
            len(grouped[route]) == 3 for route in ("graph", "owner")
        ),
        "six_unique_start_ids": len(
            {result.get("start_uuid") for result in loaded if result.get("start_uuid")}
        )
        == 6,
        "canonical_model_manifest_exact": len(loaded) == 6
        and all(
            result.get("model_manifest_sha256") == workload.model_manifest_sha256
            for result in loaded
        ),
        "all_starts_passed": len(loaded) == 6
        and all(result.get("pass") is True for result in loaded),
        "exact_graph_owner_semantics": not semantic_mismatches,
        "whole_process_tree_cpu": len(loaded) == 6
        and all(
            result.get("process_tree_cpu", {}).get("method")
            == "sampled-/proc whole-process-tree utime+stime"
            and result.get("process_tree_cpu", {}).get("observed_process_identities", 0)
            >= 2
            and result.get("process_tree_cpu", {}).get("wall_duration_ns", 0)
            >= result.get("process_tree_cpu", {}).get("load_duration_ns", 0)
            > 0
            for result in loaded
        ),
        "owner_route_identity": len(owner_identity) == 3
        and all(
            isinstance(item, dict)
            and item.get("route") == "persistent_device_model_owner"
            and item.get("host_visible_decode_epoch") is False
            and item.get("host_token_step_api") is False
            and item.get("forbidden_runtime_modules") == []
            for item in owner_identity
        ),
        "owner_exact_commit_coverage": len(grouped["owner"]) == 3
        and all(
            result.get("owner_counter_delta", {}).get("commit_events") == 8192
            and result.get("owner_counter_delta", {}).get("host_decode_steps") == 0
            and result.get("owner_counter_delta", {}).get("admission_events") == 32
            and result.get("owner_counter_delta", {}).get("aicore_calls") == 3064
            for result in grouped["owner"]
        ),
    }
    execution_pass = all(execution_checks.values())
    qualification_pass = execution_pass and all(threshold_checks.values())
    return {
        "schema_version": 1,
        "gate": "P5-GATE-persistent-decode-qualification",
        "input_sha256": {
            str(path): _sha256(path) for path in [workload_path, *paths]
        },
        "run_labels": labels,
        "aggregates": aggregates,
        "thresholds": workload.thresholds,
        "threshold_values": threshold_values,
        "threshold_checks": threshold_checks,
        "semantic_mismatches": semantic_mismatches,
        "execution_checks": execution_checks,
        "execution_pass": execution_pass,
        "qualification_pass": qualification_pass,
        "decision": (
            "P5-Persistent-Decode-Qualified"
            if qualification_pass
            else "P5-not-qualified-profile-once-before-redesign"
        ),
        "claim_boundary": (
            "Persistent Decode only. TTFT final guard and real concurrent "
            "Prefill/Decode remain P6 gates."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != 6:
        parser.error("P5 verification requires six inputs in frozen order")
    result = verify(
        [path.resolve(strict=True) for path in args.input],
        args.workload.resolve(strict=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["qualification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
