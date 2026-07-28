#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
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
from vllm_ascend_resident_epoch.abi import ABI_BYTES
from vllm_ascend_resident_epoch.scheduler import ResidentEpochScheduler
from vllm_ascend_resident_epoch.worker import ResidentEpochWorker


def _class_name(value: object) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


def _timed_call(function: Any) -> tuple[Any, int, int]:
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    value = function()
    cpu_us = (time.process_time_ns() - cpu_start) // 1000
    wall_us = (time.perf_counter_ns() - wall_start) // 1000
    return value, wall_us, cpu_us


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "host_control_wall_us",
        "python_cpu_us",
        "native_wall_us",
        "native_cpu_us",
        "cleanup_wall_us",
        "cleanup_python_cpu_us",
    )
    summary: dict[str, Any] = {"sample_count": len(samples)}
    for field in fields:
        values = [int(sample[field]) for sample in samples]
        summary[field] = {
            "min": min(values),
            "median": int(statistics.median(values)),
            "max": max(values),
        }
    return summary


def _run_sample(
    engine_core: EngineCore,
    route: str,
    repetition: int,
    expected_tokens: list[int],
    input_token_id: int,
) -> dict[str, Any]:
    prefix = f"{route}-{repetition:02d}"
    requests = [
        make_request(
            request_id=f"{prefix}-r{row}",
            client_index=repetition * 4 + row,
            max_tokens=2,
            eos_token_id=DEFAULT_EOS_TOKEN_ID,
            input_token_id=input_token_id,
        )
        for row in range(4)
    ]

    admission_wall_start = time.perf_counter_ns()
    admission_cpu_start = time.process_time_ns()
    for request in requests:
        engine_core.add_request(request)
    admission_python_cpu_us = (time.process_time_ns() - admission_cpu_start) // 1000
    admission_wall_us = (time.perf_counter_ns() - admission_wall_start) // 1000

    epoch_wall_clock_start_ns = time.time_ns()
    host_wall_start = time.perf_counter_ns()
    host_cpu_start = time.process_time_ns()
    step_wall_start = time.perf_counter_ns()
    step_cpu_start = time.process_time_ns()
    engine_outputs, model_executed = engine_core.step()
    step_python_cpu_us = (time.process_time_ns() - step_cpu_start) // 1000
    step_wall_us = (time.perf_counter_ns() - step_wall_start) // 1000
    post_wall_start = time.perf_counter_ns()
    post_cpu_start = time.process_time_ns()
    engine_core.post_step(model_executed)
    post_step_python_cpu_us = (time.process_time_ns() - post_cpu_start) // 1000
    post_step_wall_us = (time.perf_counter_ns() - post_wall_start) // 1000
    python_cpu_us = (time.process_time_ns() - host_cpu_start) // 1000
    host_control_wall_us = (time.perf_counter_ns() - host_wall_start) // 1000
    epoch_wall_clock_end_ns = time.time_ns()

    scheduler = engine_core.scheduler
    plan = getattr(scheduler, "_resident_epoch_last_plan", None)
    resident_result = getattr(scheduler, "_resident_epoch_last_result", None)
    if not isinstance(plan, ResidentEpochPlan):
        raise RuntimeError(f"{prefix}: no resident plan")
    if not isinstance(resident_result, ResidentEpochResult):
        raise RuntimeError(f"{prefix}: no resident result")

    records = output_records(engine_outputs)
    sampled = {record["request_id"]: record["tokens"] for record in records}
    expected = {request.request_id: expected_tokens[:2] for request in requests}
    state_before_cleanup = {
        request.request_id: {
            "status": request.status.name,
            "computed": request.num_computed_tokens,
            "output": request.num_output_tokens,
        }
        for request in requests
    }

    cleanup_wall_start = time.perf_counter_ns()
    cleanup_cpu_start = time.process_time_ns()
    cleanup_outputs, cleanup_model_executed = engine_core.step()
    engine_core.post_step(cleanup_model_executed)
    cleanup_python_cpu_us = (time.process_time_ns() - cleanup_cpu_start) // 1000
    cleanup_wall_us = (time.perf_counter_ns() - cleanup_wall_start) // 1000
    cleanup_records = output_records(cleanup_outputs)

    expected_abi = ABI_BYTES[route]
    checks = {
        "model_executed": model_executed is True,
        "fixed_b4_k2": plan.graph_batch_size == 4 and plan.max_steps == 2,
        "dense_active_mask": plan.active_mask == (1, 1, 1, 1),
        "tokens_match": sampled == expected,
        "accounting_match": all(
            state["computed"] == 2
            and state["output"] == 2
            and state["status"].startswith("FINISHED_")
            for state in state_before_cleanup.values()
        ),
        "one_feed_fetch": resident_result.feed_calls == 1
        and resident_result.fetch_calls == 1,
        "one_socket_round_trip": resident_result.socket_send_calls == 1
        and resident_result.socket_receive_calls == 1,
        "two_model_calls": resident_result.model_calls == 2,
        "declared_input_bytes": resident_result.declared_input_bytes
        == expected_abi["input"],
        "declared_output_bytes": resident_result.declared_output_bytes
        == expected_abi["output"],
        "native_cpu_recorded": resident_result.native_cpu_us > 0,
        "cleanup_no_model": cleanup_model_executed is False,
        "cleanup_empty": cleanup_records == [],
        "scheduler_drained": not scheduler.has_requests(),
    }
    return {
        "route": route,
        "repetition": repetition,
        "epoch_wall_clock_start_ns": epoch_wall_clock_start_ns,
        "epoch_wall_clock_end_ns": epoch_wall_clock_end_ns,
        "admission_calls": len(requests),
        "admission_wall_us": admission_wall_us,
        "admission_python_cpu_us": admission_python_cpu_us,
        "engine_core_step_calls": 1,
        "post_step_calls": 1,
        "socket_send_calls": resident_result.socket_send_calls,
        "socket_receive_calls": resident_result.socket_receive_calls,
        "feed_calls": resident_result.feed_calls,
        "fetch_calls": resident_result.fetch_calls,
        "model_calls": resident_result.model_calls,
        "step_wall_us": step_wall_us,
        "step_python_cpu_us": step_python_cpu_us,
        "post_step_wall_us": post_step_wall_us,
        "post_step_python_cpu_us": post_step_python_cpu_us,
        "host_control_wall_us": host_control_wall_us,
        "python_cpu_us": python_cpu_us,
        "native_wall_us": resident_result.wall_us,
        "native_cpu_us": resident_result.native_cpu_us,
        "declared_input_bytes": resident_result.declared_input_bytes,
        "declared_output_bytes": resident_result.declared_output_bytes,
        "declared_total_bytes": resident_result.declared_input_bytes
        + resident_result.declared_output_bytes,
        "row_generations": list(resident_result.row_generations),
        "sampled_tokens": sampled,
        "request_state": state_before_cleanup,
        "cleanup_step_calls": 1,
        "cleanup_post_step_calls": 1,
        "cleanup_wall_us": cleanup_wall_us,
        "cleanup_python_cpu_us": cleanup_python_cpu_us,
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
    parser.add_argument("--route", choices=tuple(ABI_BYTES), required=True)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--input-token-id", type=int, default=DEFAULT_INPUT_TOKEN_ID)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    result: dict[str, Any] = {
        "gate": "Attempt 74 minimal ABI EngineCore epoch benchmark",
        "schema_version": 1,
        "route": args.route,
        "repetitions": args.repetitions,
        "pass": False,
        "samples": [],
        "artifacts": {},
    }
    engine_core: EngineCore | None = None
    try:
        model_config = args.model_config.resolve(strict=True)
        baseline_result = args.baseline_result.resolve(strict=True)
        expected_tokens = load_expected_tokens(baseline_result)[2]
        vllm_config = make_engine_vllm_config(model_config)

        init_wall_start = time.perf_counter_ns()
        init_cpu_start = time.process_time_ns()
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=UniProcExecutor,
            log_stats=True,
        )
        result["engine_core_init_python_cpu_us"] = (
            time.process_time_ns() - init_cpu_start
        ) // 1000
        result["engine_core_init_wall_us"] = (
            time.perf_counter_ns() - init_wall_start
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
        if worker.warmup_output is None:
            raise RuntimeError("EngineCore initialization skipped resident warmup")

        warmup = worker.warmup_output
        result["warmup"] = {
            "status": warmup.status,
            "model_calls": warmup.model_calls,
            "feed_calls": warmup.feed_calls,
            "fetch_calls": warmup.fetch_calls,
            "wall_us": warmup.wall_us,
            "native_cpu_us": warmup.native_cpu_us,
            "declared_input_bytes": warmup.declared_input_bytes,
            "declared_output_bytes": warmup.declared_output_bytes,
        }
        result["distributed_initialized_during_engine"] = (
            torch.distributed.is_initialized()
        )
        for repetition in range(args.repetitions):
            result["samples"].append(
                _run_sample(
                    engine_core,
                    args.route,
                    repetition,
                    expected_tokens,
                    args.input_token_id,
                )
            )
        result["summary"] = _summarize(result["samples"])
        result["artifacts"] = {
            "baseline_result": str(baseline_result),
            "baseline_sha256": sha256(baseline_result),
            "native_server": os.environ.get("VLLM_ASCEND_RESIDENT_EPOCH_SERVER"),
            "func_config": os.environ.get(
                "VLLM_ASCEND_RESIDENT_EPOCH_FUNC_CONFIG"
            ),
            "air": os.environ.get("VLLM_ASCEND_RESIDENT_EPOCH_AIR"),
            "external_weights": os.environ.get(
                "VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS"
            ),
        }
        result["pass"] = (
            result["distributed_initialized_during_engine"]
            and all(sample["pass"] for sample in result["samples"])
        )
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
