#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from run_engine_core_native import make_engine_vllm_config
from run_scheduler_native import (
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_INPUT_TOKEN_ID,
    load_expected_tokens,
    make_request,
    output_records,
    sha256,
)
from vllm_ascend_resident_epoch.contract import ResidentEpochPlan, ResidentEpochResult
from vllm_ascend_resident_epoch.scheduler import ResidentEpochScheduler
from vllm_ascend_resident_epoch.worker import ResidentEpochWorker


FIRST_SERVICE_EPOCH_LIMIT_US = 10_000_000


def _class_name(value: object) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


def _request_state(request: Any) -> dict[str, Any]:
    return {
        "status": request.status.name,
        "num_computed_tokens": request.num_computed_tokens,
        "num_output_tokens": request.num_output_tokens,
        "all_token_ids": list(request.all_token_ids),
    }


def _execute_epoch(
    engine_core: EngineCore,
    name: str,
    requests: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter_ns()
    engine_outputs, model_executed = engine_core.step()
    wall_us = (time.perf_counter_ns() - start) // 1000
    engine_core.post_step(model_executed)

    scheduler = engine_core.scheduler
    plan = getattr(scheduler, "_resident_epoch_last_plan", None)
    result = getattr(scheduler, "_resident_epoch_last_result", None)
    if not isinstance(plan, ResidentEpochPlan):
        raise RuntimeError(
            f"{name}: no resident plan, rejection="
            f"{getattr(scheduler, '_resident_epoch_last_rejection', None)}"
        )
    if not isinstance(result, ResidentEpochResult):
        raise RuntimeError(f"{name}: no resident execution result")

    records = output_records(engine_outputs)
    sampled = {record["request_id"]: record["tokens"] for record in records}
    return {
        "name": name,
        "model_executed": model_executed,
        "engine_core_step_wall_us": wall_us,
        "native_wall_us": result.wall_us,
        "model_calls": result.model_calls,
        "feed_calls": result.feed_calls,
        "fetch_calls": result.fetch_calls,
        "computed_steps": result.computed_steps,
        "result_row_generations": list(result.row_generations),
        "sampled_tokens": sampled,
        "engine_outputs": records,
        "plan": {
            "max_steps": plan.max_steps,
            "active_mask": list(plan.active_mask),
            "row_generations": list(plan.row_generations),
            "requests": {
                request.req_id: {
                    "row": request.row,
                    "generation": request.generation,
                    "token_id": request.token_id,
                    "position": request.position,
                    "sequence_length": request.sequence_length,
                    "device_block_ids": list(request.device_block_ids),
                }
                for request in plan.requests
            },
        },
        "request_state": {
            request_id: _request_state(request)
            for request_id, request in requests.items()
        },
    }


def run_dynamic_trace(
    engine_core: EngineCore,
    expected_tokens: list[int],
    input_token_id: int,
) -> dict[str, Any]:
    if len(expected_tokens) < 6:
        raise ValueError("the dynamic trace requires at least six baseline tokens")

    requests: dict[str, Any] = {}
    requests["A"] = make_request(
        request_id="A",
        client_index=0,
        max_tokens=6,
        eos_token_id=DEFAULT_EOS_TOKEN_ID,
        input_token_id=input_token_id,
    )
    engine_core.add_request(requests["A"])
    epochs = [_execute_epoch(engine_core, "epoch-1-a", requests)]

    requests["B"] = make_request(
        request_id="B",
        client_index=1,
        max_tokens=2,
        eos_token_id=DEFAULT_EOS_TOKEN_ID,
        input_token_id=input_token_id,
    )
    engine_core.add_request(requests["B"])
    epochs.append(_execute_epoch(engine_core, "epoch-2-a-b", requests))

    requests["C"] = make_request(
        request_id="C",
        client_index=2,
        max_tokens=2,
        eos_token_id=DEFAULT_EOS_TOKEN_ID,
        input_token_id=input_token_id,
    )
    engine_core.add_request(requests["C"])
    epochs.append(_execute_epoch(engine_core, "epoch-3-a-c", requests))

    cleanup_outputs, cleanup_model_executed = engine_core.step()
    engine_core.post_step(cleanup_model_executed)
    cleanup_records = output_records(cleanup_outputs)

    observed = {
        "A": (
            epochs[0]["sampled_tokens"].get("A", [])
            + epochs[1]["sampled_tokens"].get("A", [])
            + epochs[2]["sampled_tokens"].get("A", [])
        ),
        "B": epochs[1]["sampled_tokens"].get("B", []),
        "C": epochs[2]["sampled_tokens"].get("C", []),
    }
    expected = {
        "A": expected_tokens[:6],
        "B": expected_tokens[:2],
        "C": expected_tokens[:2],
    }
    expected_epoch_tokens = [
        {"A": expected_tokens[:2]},
        {"A": expected_tokens[2:4], "B": expected_tokens[:2]},
        {"A": expected_tokens[4:6], "C": expected_tokens[:2]},
    ]
    expected_plan_requests = [
        {"A": (0, 1, 0, input_token_id)},
        {"A": (0, 1, 2, expected_tokens[1]), "B": (1, 2, 0, input_token_id)},
        {"A": (0, 1, 4, expected_tokens[3]), "C": (1, 3, 0, input_token_id)},
    ]

    epoch_checks: list[dict[str, bool]] = []
    for index, epoch in enumerate(epochs):
        actual_plan = {
            request_id: (
                request["row"],
                request["generation"],
                request["position"],
                request["token_id"],
            )
            for request_id, request in epoch["plan"]["requests"].items()
        }
        expected_active = [1, 0, 0, 0] if index == 0 else [1, 1, 0, 0]
        expected_generations = [1, 0, 0, 0]
        if index == 1:
            expected_generations = [1, 2, 0, 0]
        elif index == 2:
            expected_generations = [1, 3, 0, 0]
        checks = {
            "model_executed": epoch["model_executed"] is True,
            "two_step_epoch": epoch["plan"]["max_steps"] == 2,
            "one_feed": epoch["feed_calls"] == 1,
            "one_fetch": epoch["fetch_calls"] == 1,
            "two_model_calls": epoch["model_calls"] == 2,
            "active_mask": epoch["plan"]["active_mask"] == expected_active,
            "stable_rows_and_generations": actual_plan
            == expected_plan_requests[index],
            "generation_acknowledgement": epoch["result_row_generations"]
            == expected_generations,
            "epoch_tokens": epoch["sampled_tokens"] == expected_epoch_tokens[index],
        }
        epoch["checks"] = checks
        epoch["pass"] = all(checks.values())
        epoch_checks.append(checks)

    scheduler = engine_core.scheduler
    final_states = {
        request_id: _request_state(request)
        for request_id, request in requests.items()
    }
    checks = {
        "three_device_epochs": len(epochs) == 3
        and all(epoch["pass"] for epoch in epochs),
        "tokens_match_frozen_baseline": observed == expected,
        "a_spans_three_epochs": all("A" in epoch["sampled_tokens"] for epoch in epochs),
        "row_zero_generation_one_stable": all(
            epoch["plan"]["requests"]["A"]["row"] == 0
            and epoch["plan"]["requests"]["A"]["generation"] == 1
            for epoch in epochs
        ),
        "row_one_replaced": epochs[1]["plan"]["requests"]["B"]["generation"] == 2
        and epochs[2]["plan"]["requests"]["C"]["generation"] == 3,
        "final_accounting": {
            request_id: (
                state["num_computed_tokens"],
                state["num_output_tokens"],
            )
            for request_id, state in final_states.items()
        }
        == {"A": (6, 6), "B": (2, 2), "C": (2, 2)},
        "all_requests_finished": all(
            state["status"].startswith("FINISHED_") for state in final_states.values()
        ),
        "cleanup_model_not_executed": cleanup_model_executed is False,
        "cleanup_outputs_empty": cleanup_records == [],
        "scheduler_drained": not scheduler.has_requests(),
    }
    return {
        "epochs": epochs,
        "observed_tokens": observed,
        "expected_tokens": expected,
        "final_request_state": final_states,
        "cleanup": {
            "model_executed": cleanup_model_executed,
            "engine_outputs": cleanup_records,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-token-id", type=int, default=DEFAULT_INPUT_TOKEN_ID)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "gate": "Attempt 73 multi-epoch residency, cohort update, and warmup",
        "schema_version": 1,
        "pass": False,
        "support_boundary": {
            "model": "Qwen2.5-7B-Instruct",
            "epoch_steps": 2,
            "logical_capacity": 8,
            "trace": ["A", "A+B", "A+C"],
            "sampling": "greedy",
            "prompt_tokens": 1,
            "tp": 1,
            "pp": 1,
        },
        "artifacts": {},
    }
    engine_core: EngineCore | None = None
    try:
        model_config = args.model_config.resolve(strict=True)
        baseline_result = args.baseline_result.resolve(strict=True)
        expected_tokens = load_expected_tokens(baseline_result)[8]
        vllm_config = make_engine_vllm_config(model_config)

        init_start = time.perf_counter_ns()
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=UniProcExecutor,
            log_stats=True,
        )
        result["engine_core_init_wall_us"] = (
            time.perf_counter_ns() - init_start
        ) // 1000

        worker = engine_core.model_executor.driver_worker.worker
        result["resolved_classes"] = {
            "scheduler": _class_name(engine_core.scheduler),
            "executor": _class_name(engine_core.model_executor),
            "worker": _class_name(worker),
        }
        if not isinstance(engine_core.scheduler, ResidentEpochScheduler):
            raise RuntimeError("EngineCore did not resolve ResidentEpochScheduler")
        if not isinstance(engine_core.model_executor, UniProcExecutor):
            raise RuntimeError("EngineCore did not use UniProcExecutor")
        if not isinstance(worker, ResidentEpochWorker):
            raise RuntimeError("EngineCore did not resolve ResidentEpochWorker")
        if worker.resident_config.max_steps != 2:
            raise RuntimeError("Attempt 73 requires a two-step resident epoch")
        if worker.resident_config.logical_capacity != 8:
            raise RuntimeError("Attempt 73 requires logical capacity eight")
        if worker.warmup_output is None:
            raise RuntimeError("EngineCore initialization skipped resident warmup")

        result["distributed_initialized_during_engine"] = (
            torch.distributed.is_initialized()
        )
        result["warmup"] = {
            "status": worker.warmup_output.status,
            "model_calls": worker.warmup_output.model_calls,
            "feed_calls": worker.warmup_output.feed_calls,
            "fetch_calls": worker.warmup_output.fetch_calls,
            "wall_us": worker.warmup_output.wall_us,
        }
        result["trace"] = run_dynamic_trace(
            engine_core,
            expected_tokens,
            args.input_token_id,
        )
        first_service_wall_us = result["trace"]["epochs"][0][
            "engine_core_step_wall_us"
        ]
        result["checks"] = {
            "distributed_initialized": result[
                "distributed_initialized_during_engine"
            ],
            "warmup_one_complete_step": result["warmup"]
            == {
                "status": 0,
                "model_calls": 1,
                "feed_calls": 1,
                "fetch_calls": 1,
                "wall_us": result["warmup"]["wall_us"],
            }
            and result["warmup"]["wall_us"] > 0,
            "first_service_epoch_below_limit": first_service_wall_us
            < FIRST_SERVICE_EPOCH_LIMIT_US,
            "dynamic_trace_passed": result["trace"]["pass"],
        }
        result["artifacts"] = {
            "baseline_result": str(baseline_result),
            "baseline_sha256": sha256(baseline_result),
            "native_server": os.environ.get("VLLM_ASCEND_RESIDENT_EPOCH_SERVER"),
            "backend_factory": os.environ.get(
                "VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY"
            ),
            "air": os.environ.get("VLLM_ASCEND_RESIDENT_EPOCH_AIR"),
            "tiling": os.environ.get("VLLM_ASCEND_RESIDENT_EPOCH_TILING"),
            "external_weights": os.environ.get(
                "VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS"
            ),
        }
        result["pass"] = all(result["checks"].values())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if engine_core is not None:
            try:
                engine_core.shutdown()
            except Exception as exc:
                result["shutdown_error"] = f"{type(exc).__name__}: {exc}"
                result["pass"] = False
        elif torch.distributed.is_initialized():
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        result["distributed_initialized_after_shutdown"] = (
            torch.distributed.is_initialized()
        )
        if result["distributed_initialized_after_shutdown"]:
            result["pass"] = False

    _write_result(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
