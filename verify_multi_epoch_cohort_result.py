#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_GATE = "Attempt 74 minimal ABI multi-epoch residency"
EXPECTED_CLASSES = {
    "scheduler": "vllm_ascend_resident_epoch.scheduler.ResidentEpochScheduler",
    "executor": "vllm.v1.executor.uniproc_executor.UniProcExecutor",
    "worker": "vllm_ascend_resident_epoch.worker.ResidentEpochWorker",
}
EXPECTED_SUPPORT = {
    "model": "Qwen2.5-7B-Instruct",
    "epoch_steps": 2,
    "logical_capacity": 8,
    "trace": ["A", "A+B", "A+C"],
    "sampling": "greedy",
    "prompt_tokens": 1,
    "tp": 1,
    "pp": 1,
    "host_udf_abi": "8-input/2-output",
}
EXPECTED_TOKENS = [17728, 374, 264, 4185, 19734, 24844]
EXPECTED_EPOCHS = [
    {
        "name": "epoch-1-a",
        "active_mask": [1, 0, 0, 0],
        "row_generations": [1, 0, 0, 0],
        "sampled_tokens": {"A": EXPECTED_TOKENS[:2]},
        "requests": {"A": (0, 1, 0, 11690)},
    },
    {
        "name": "epoch-2-a-b",
        "active_mask": [1, 1, 0, 0],
        "row_generations": [1, 2, 0, 0],
        "sampled_tokens": {
            "A": EXPECTED_TOKENS[2:4],
            "B": EXPECTED_TOKENS[:2],
        },
        "requests": {
            "A": (0, 1, 2, EXPECTED_TOKENS[1]),
            "B": (1, 2, 0, 11690),
        },
    },
    {
        "name": "epoch-3-a-c",
        "active_mask": [1, 1, 0, 0],
        "row_generations": [1, 3, 0, 0],
        "sampled_tokens": {
            "A": EXPECTED_TOKENS[4:6],
            "C": EXPECTED_TOKENS[:2],
        },
        "requests": {
            "A": (0, 1, 4, EXPECTED_TOKENS[3]),
            "C": (1, 3, 0, 11690),
        },
    },
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_all_checks(value: Any, name: str) -> None:
    _require(isinstance(value, dict) and bool(value), f"{name}: missing checks")
    failed = sorted(key for key, passed in value.items() if passed is not True)
    _require(not failed, f"{name}: failed checks: {failed}")


def validate_result(data: dict[str, Any], *, require_artifacts: bool) -> dict[str, Any]:
    _require(data.get("gate") == EXPECTED_GATE, "unexpected gate name")
    _require(data.get("schema_version") == 1, "unexpected schema version")
    _require(data.get("pass") is True, "runner did not pass")
    _require("error" not in data, "runner recorded an execution error")
    _require("shutdown_error" not in data, "runner recorded a shutdown error")
    _require(data.get("support_boundary") == EXPECTED_SUPPORT, "support changed")
    _require(data.get("resolved_classes") == EXPECTED_CLASSES, "classes changed")
    _require(
        data.get("distributed_initialized_during_engine") is True,
        "distributed environment was not initialized",
    )
    _require(
        data.get("distributed_initialized_after_shutdown") is False,
        "distributed environment survived shutdown",
    )
    _require(isinstance(data.get("engine_core_init_wall_us"), int), "missing init time")

    warmup = data.get("warmup")
    _require(isinstance(warmup, dict), "missing warmup result")
    _require(warmup.get("status") == 0, "warmup device status is not zero")
    _require(warmup.get("model_calls") == 1, "warmup model-call count is not one")
    _require(warmup.get("feed_calls") == 1, "warmup Feed count is not one")
    _require(warmup.get("fetch_calls") == 1, "warmup Fetch count is not one")
    _require(
        isinstance(warmup.get("wall_us"), int) and warmup["wall_us"] > 0,
        "warmup wall time is invalid",
    )
    _require(
        isinstance(warmup.get("native_cpu_us"), int)
        and warmup["native_cpu_us"] > 0,
        "warmup native CPU time is invalid",
    )
    _require(
        warmup.get("declared_input_bytes") == 260
        and warmup.get("declared_output_bytes") == 368,
        "warmup did not use the minimal ABI",
    )
    _require_all_checks(data.get("checks"), "top-level")

    artifacts = data.get("artifacts")
    _require(isinstance(artifacts, dict), "missing artifact manifest")
    _require(
        artifacts.get("backend_factory")
        == "vllm_ascend_resident_epoch.sidecar_backend:create_sidecar_engine",
        "runner did not use the DataFlow sidecar",
    )
    baseline_hash = artifacts.get("baseline_sha256")
    _require(
        isinstance(baseline_hash, str) and len(baseline_hash) == 64,
        "invalid baseline hash",
    )
    if require_artifacts:
        for key in (
            "baseline_result",
            "native_server",
            "air",
            "tiling",
            "external_weights",
        ):
            value = artifacts.get(key)
            _require(isinstance(value, str) and bool(value), f"missing artifact {key}")
            _require(Path(value).exists(), f"artifact does not exist: {key}={value}")

    trace = data.get("trace")
    _require(isinstance(trace, dict) and trace.get("pass") is True, "trace failed")
    _require_all_checks(trace.get("checks"), "trace")
    _require(
        trace.get("observed_tokens")
        == {
            "A": EXPECTED_TOKENS,
            "B": EXPECTED_TOKENS[:2],
            "C": EXPECTED_TOKENS[:2],
        },
        "cross-epoch tokens do not match the frozen baseline",
    )
    _require(trace.get("observed_tokens") == trace.get("expected_tokens"), "token self-check failed")

    epochs = trace.get("epochs")
    _require(isinstance(epochs, list) and len(epochs) == 3, "expected three epochs")
    total_feed = 1
    total_fetch = 1
    total_model_calls = 1
    for epoch, expected in zip(epochs, EXPECTED_EPOCHS, strict=True):
        name = expected["name"]
        _require(epoch.get("name") == name, f"{name}: epoch order changed")
        _require(epoch.get("pass") is True, f"{name}: runner marked epoch failed")
        _require_all_checks(epoch.get("checks"), name)
        _require(epoch.get("model_executed") is True, f"{name}: model not executed")
        _require(epoch.get("feed_calls") == 1, f"{name}: Feed count is not one")
        _require(epoch.get("fetch_calls") == 1, f"{name}: Fetch count is not one")
        _require(epoch.get("model_calls") == 2, f"{name}: model-call count is not two")
        _require(
            epoch.get("socket_send_calls") == 1
            and epoch.get("socket_receive_calls") == 1,
            f"{name}: socket round-trip count changed",
        )
        _require(
            epoch.get("declared_input_bytes") == 260
            and epoch.get("declared_output_bytes") == 368
            and epoch.get("declared_total_bytes") == 628,
            f"{name}: minimal ABI byte count changed",
        )
        _require(
            isinstance(epoch.get("python_cpu_us"), int)
            and epoch["python_cpu_us"] >= 0,
            f"{name}: missing Python CPU time",
        )
        _require(
            isinstance(epoch.get("native_cpu_us"), int)
            and epoch["native_cpu_us"] > 0,
            f"{name}: missing native CPU time",
        )
        _require(epoch.get("sampled_tokens") == expected["sampled_tokens"], f"{name}: tokens changed")
        plan = epoch.get("plan")
        _require(isinstance(plan, dict), f"{name}: missing plan")
        _require(plan.get("max_steps") == 2, f"{name}: epoch budget changed")
        _require(plan.get("active_mask") == expected["active_mask"], f"{name}: active mask changed")
        _require(plan.get("row_generations") == expected["row_generations"], f"{name}: plan generations changed")
        _require(epoch.get("result_row_generations") == expected["row_generations"], f"{name}: generation acknowledgement changed")
        requests = plan.get("requests")
        _require(isinstance(requests, dict), f"{name}: missing request plans")
        actual_requests = {
            request_id: (
                request.get("row"),
                request.get("generation"),
                request.get("position"),
                request.get("token_id"),
            )
            for request_id, request in requests.items()
        }
        _require(actual_requests == expected["requests"], f"{name}: row continuity changed")
        total_feed += epoch["feed_calls"]
        total_fetch += epoch["fetch_calls"]
        total_model_calls += epoch["model_calls"]

    _require(
        epochs[0].get("engine_core_step_wall_us", 10_000_000) < 10_000_000,
        "first service epoch still contains lazy-load latency",
    )
    final_state = trace.get("final_request_state")
    _require(isinstance(final_state, dict), "missing final request state")
    _require(
        {
            request_id: (
                state.get("num_computed_tokens"),
                state.get("num_output_tokens"),
            )
            for request_id, state in final_state.items()
        }
        == {"A": (6, 6), "B": (2, 2), "C": (2, 2)},
        "final scheduler accounting changed",
    )
    _require(
        all(state.get("status", "").startswith("FINISHED_") for state in final_state.values()),
        "not all requests finished",
    )
    cleanup = trace.get("cleanup")
    _require(
        cleanup == {"model_executed": False, "engine_outputs": []},
        "cleanup executed the model or emitted output",
    )
    return {
        "pass": True,
        "warmup_feed_calls": 1,
        "warmup_fetch_calls": 1,
        "service_epochs": 3,
        "service_feed_calls": total_feed - 1,
        "service_fetch_calls": total_fetch - 1,
        "total_model_calls": total_model_calls,
        "first_service_epoch_wall_us": epochs[0]["engine_core_step_wall_us"],
        "resolved_classes": data["resolved_classes"],
        "distributed_destroyed": True,
        "declared_input_bytes_per_epoch": 260,
        "declared_output_bytes_per_epoch": 368,
        "declared_total_bytes_per_epoch": 628,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.result.read_text(encoding="utf-8"))
        summary = validate_result(data, require_artifacts=True)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"MULTI_EPOCH_RESULT_INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
