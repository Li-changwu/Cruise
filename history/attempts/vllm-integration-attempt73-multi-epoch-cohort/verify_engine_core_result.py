#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_GATE = "Attempt 72 EngineCore plus UniProcExecutor resident epoch"
EXPECTED_CLASSES = {
    "scheduler": (
        "vllm_ascend_resident_epoch.scheduler.ResidentEpochScheduler"
    ),
    "executor": "vllm.v1.executor.uniproc_executor.UniProcExecutor",
    "worker": "vllm_ascend_resident_epoch.worker.ResidentEpochWorker",
}
EXPECTED_SUPPORT = {
    "model": "Qwen2.5-7B-Instruct",
    "batch_sizes": [1, 2, 4],
    "max_steps": [1, 2, 4, 8],
    "sampling": "greedy",
    "prompt_tokens": 1,
    "tp": 1,
    "pp": 1,
}
REQUIRED_CHECKS = {
    "engine_core_executed_model",
    "fixed_b4_plan",
    "dense_active_mask",
    "tokens_match_frozen_baseline",
    "one_feed",
    "one_fetch",
    "model_calls_match",
    "computed_steps_match",
    "scheduler_accounting_match",
    "requests_finished",
    "cleanup_model_not_executed",
    "cleanup_outputs_empty",
    "scheduler_drained",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_cases() -> dict[str, tuple[int, int]]:
    cases = {
        f"b{batch_size}-k{max_steps}": (batch_size, max_steps)
        for batch_size in (1, 2, 4)
        for max_steps in (1, 2, 4, 8)
    }
    cases["b4-independent-eos"] = (4, 4)
    return cases


def validate_result(data: dict[str, Any], *, require_artifacts: bool) -> dict[str, Any]:
    _require(data.get("gate") == EXPECTED_GATE, "unexpected gate name")
    _require(data.get("schema_version") == 1, "unexpected schema version")
    _require(data.get("pass") is True, "runner did not pass")
    _require("error" not in data, "runner recorded an execution error")
    _require("shutdown_error" not in data, "runner recorded a shutdown error")
    _require(data.get("support_boundary") == EXPECTED_SUPPORT, "support boundary changed")
    _require(data.get("resolved_classes") == EXPECTED_CLASSES, "resolved classes changed")
    _require(
        data.get("distributed_initialized_during_engine") is True,
        "distributed environment was not initialized",
    )
    _require(
        data.get("distributed_initialized_after_shutdown") is False,
        "distributed environment survived EngineCore shutdown",
    )

    artifacts = data.get("artifacts")
    _require(isinstance(artifacts, dict), "missing artifact manifest")
    _require(
        artifacts.get("backend_factory")
        == "vllm_ascend_resident_epoch.sidecar_backend:create_sidecar_engine",
        "EngineCore did not use the DataFlow sidecar backend",
    )
    baseline_sha256 = artifacts.get("baseline_sha256")
    _require(
        isinstance(baseline_sha256, str) and len(baseline_sha256) == 64,
        "invalid baseline hash",
    )
    if require_artifacts:
        for key in (
            "baseline_result",
            "native_library",
            "native_server",
            "air",
            "tiling",
            "external_weights",
        ):
            value = artifacts.get(key)
            _require(isinstance(value, str) and bool(value), f"missing artifact {key}")
            _require(Path(value).exists(), f"artifact does not exist: {key}={value}")

    raw_cases = data.get("cases")
    _require(isinstance(raw_cases, list), "cases must be a list")
    cases = {case.get("name"): case for case in raw_cases if isinstance(case, dict)}
    expected = expected_cases()
    _require(len(raw_cases) == len(expected), "expected exactly 13 cases")
    _require(set(cases) == set(expected), "case matrix is incomplete or duplicated")

    total_device_epochs = 0
    total_model_calls = 0
    for name, (batch_size, max_steps) in expected.items():
        case = cases[name]
        _require(case.get("pass") is True, f"{name}: case did not pass")
        _require(case.get("batch_size") == batch_size, f"{name}: batch size changed")
        _require(case.get("max_steps") == max_steps, f"{name}: max steps changed")
        _require(case.get("feed_calls") == 1, f"{name}: Feed count is not one")
        _require(case.get("fetch_calls") == 1, f"{name}: Fetch count is not one")
        _require(case.get("cleanup_engine_outputs") == [], f"{name}: cleanup emitted output")

        checks = case.get("checks")
        _require(isinstance(checks, dict), f"{name}: missing checks")
        _require(REQUIRED_CHECKS <= set(checks), f"{name}: required checks missing")
        _require(all(checks.values()), f"{name}: one or more checks failed")

        plan = case.get("plan")
        _require(isinstance(plan, dict), f"{name}: missing plan")
        _require(plan.get("graph_batch_size") == 4, f"{name}: graph is not B=4")
        _require(
            plan.get("active_mask") == [1] * batch_size + [0] * (4 - batch_size),
            f"{name}: active mask changed",
        )
        _require(
            len(case.get("engine_outputs", [])) == batch_size,
            f"{name}: EngineCore output count changed",
        )

        computed_steps = case.get("computed_steps")
        _require(isinstance(computed_steps, dict), f"{name}: missing computed steps")
        expected_steps = (
            sorted(computed_steps.values()) == [1, 1, 4, 4]
            if name == "b4-independent-eos"
            else set(computed_steps.values()) == {max_steps}
        )
        _require(expected_steps, f"{name}: computed-step accounting changed")
        _require(
            case.get("model_calls") == max(computed_steps.values()),
            f"{name}: model-call count changed",
        )
        total_device_epochs += 1
        total_model_calls += case["model_calls"]

    return {
        "pass": True,
        "case_count": len(cases),
        "device_epochs": total_device_epochs,
        "feed_calls": total_device_epochs,
        "fetch_calls": total_device_epochs,
        "model_calls": total_model_calls,
        "resolved_classes": data["resolved_classes"],
        "distributed_destroyed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.result.read_text(encoding="utf-8"))
        summary = validate_result(data, require_artifacts=True)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ENGINE_CORE_RESULT_INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
