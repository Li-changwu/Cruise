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

from run_scheduler_native import (
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_INPUT_TOKEN_ID,
    GRAPH_BATCH_SIZE,
    load_expected_tokens,
    make_request,
    make_vllm_config,
    output_records,
    sha256,
)
from vllm_ascend_resident_epoch.contract import ResidentEpochPlan, ResidentEpochResult
from vllm_ascend_resident_epoch.scheduler import ResidentEpochScheduler
from vllm_ascend_resident_epoch.worker import (
    RESIDENT_WORKER_QUALNAME,
    ResidentEpochWorker,
)


SCHEDULER_QUALNAME = (
    "vllm_ascend_resident_epoch.scheduler.ResidentEpochScheduler"
)


def make_engine_vllm_config(model: Path):
    config = make_vllm_config(model)
    config.scheduler_config.scheduler_cls = SCHEDULER_QUALNAME
    config.parallel_config.worker_cls = RESIDENT_WORKER_QUALNAME
    config.parallel_config.distributed_executor_backend = "uni"
    return config


def run_case(
    engine_core: EngineCore,
    name: str,
    batch_size: int,
    max_steps: int,
    expected_tokens: list[int],
    eos_token_ids: list[int],
    input_token_id: int,
) -> dict[str, Any]:
    requests = [
        make_request(
            request_id=f"{name}-r{row}",
            client_index=row,
            max_tokens=max_steps,
            eos_token_id=eos_token_ids[row],
            input_token_id=input_token_id,
        )
        for row in range(batch_size)
    ]
    for request in requests:
        engine_core.add_request(request)

    step_start = time.perf_counter_ns()
    engine_outputs, model_executed = engine_core.step()
    step_wall_us = (time.perf_counter_ns() - step_start) // 1000
    engine_core.post_step(model_executed)

    scheduler = engine_core.scheduler
    plan = getattr(scheduler, "_resident_epoch_last_plan", None)
    result = getattr(scheduler, "_resident_epoch_last_result", None)
    if not isinstance(plan, ResidentEpochPlan):
        raise RuntimeError(f"{name}: EngineCore did not execute a resident plan")
    if not isinstance(result, ResidentEpochResult):
        raise RuntimeError(f"{name}: EngineCore did not commit a resident result")

    records = output_records(engine_outputs)
    sampled = {record["request_id"]: record["tokens"] for record in records}
    expected_by_request = {
        request.request_id: (
            expected_tokens[:1]
            if eos_token_ids[row] == expected_tokens[0]
            else expected_tokens
        )
        for row, request in enumerate(requests)
    }
    cleanup_outputs, cleanup_model_executed = engine_core.step()
    engine_core.post_step(cleanup_model_executed)
    cleanup_records = output_records(cleanup_outputs)
    checks = {
        "engine_core_executed_model": model_executed,
        "fixed_b4_plan": plan.graph_batch_size == GRAPH_BATCH_SIZE,
        "dense_active_mask": plan.active_mask
        == (1,) * batch_size + (0,) * (GRAPH_BATCH_SIZE - batch_size),
        "tokens_match_frozen_baseline": sampled == expected_by_request,
        "one_feed": result.feed_calls == 1,
        "one_fetch": result.fetch_calls == 1,
        "model_calls_match": result.model_calls
        == max(len(tokens) for tokens in expected_by_request.values()),
        "computed_steps_match": result.computed_steps
        == {
            request_id: len(tokens)
            for request_id, tokens in expected_by_request.items()
        },
        "scheduler_accounting_match": all(
            request.num_computed_tokens
            == len(expected_by_request[request.request_id])
            and request.num_output_tokens
            == len(expected_by_request[request.request_id])
            for request in requests
        ),
        "requests_finished": all(
            request.status.name.startswith("FINISHED_") for request in requests
        ),
        "cleanup_model_not_executed": not cleanup_model_executed,
        "cleanup_outputs_empty": not cleanup_records,
        "scheduler_drained": not scheduler.has_requests(),
    }
    return {
        "name": name,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "eos_token_ids": eos_token_ids,
        "plan": {
            "graph_batch_size": plan.graph_batch_size,
            "active_mask": list(plan.active_mask),
            "scheduler_block_ids": {
                request.req_id: list(request.scheduler_block_ids)
                for request in plan.requests
            },
            "device_block_ids": {
                request.req_id: list(request.device_block_ids)
                for request in plan.requests
            },
        },
        "sampled_tokens": sampled,
        "expected_tokens": expected_by_request,
        "computed_steps": result.computed_steps,
        "model_calls": result.model_calls,
        "feed_calls": result.feed_calls,
        "fetch_calls": result.fetch_calls,
        "native_wall_us": result.wall_us,
        "engine_core_step_wall_us": step_wall_us,
        "request_status": {
            request.request_id: request.status.name for request in requests
        },
        "engine_outputs": records,
        "cleanup_engine_outputs": cleanup_records,
        "checks": checks,
        "pass": all(checks.values()),
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
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
        "gate": "Attempt 72 EngineCore plus UniProcExecutor resident epoch",
        "schema_version": 1,
        "pass": False,
        "support_boundary": {
            "model": "Qwen2.5-7B-Instruct",
            "batch_sizes": [1, 2, 4],
            "max_steps": [1, 2, 4, 8],
            "sampling": "greedy",
            "prompt_tokens": 1,
            "tp": 1,
            "pp": 1,
        },
        "artifacts": {},
        "cases": [],
    }
    engine_core: EngineCore | None = None
    try:
        model_config = args.model_config.resolve(strict=True)
        baseline_result = args.baseline_result.resolve(strict=True)
        expected = load_expected_tokens(baseline_result)
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
            "scheduler": type(engine_core.scheduler).__module__
            + "."
            + type(engine_core.scheduler).__qualname__,
            "executor": type(engine_core.model_executor).__module__
            + "."
            + type(engine_core.model_executor).__qualname__,
            "worker": type(worker).__module__ + "." + type(worker).__qualname__,
        }
        if not isinstance(engine_core.scheduler, ResidentEpochScheduler):
            raise RuntimeError("EngineCore did not resolve ResidentEpochScheduler")
        if not isinstance(engine_core.model_executor, UniProcExecutor):
            raise RuntimeError("EngineCore did not use UniProcExecutor")
        if not isinstance(worker, ResidentEpochWorker):
            raise RuntimeError("UniProcExecutor did not resolve ResidentEpochWorker")
        result["distributed_initialized_during_engine"] = (
            torch.distributed.is_initialized()
        )

        for batch_size in (1, 2, 4):
            for max_steps in (1, 2, 4, 8):
                result["cases"].append(
                    run_case(
                        engine_core=engine_core,
                        name=f"b{batch_size}-k{max_steps}",
                        batch_size=batch_size,
                        max_steps=max_steps,
                        expected_tokens=expected[max_steps],
                        eos_token_ids=[DEFAULT_EOS_TOKEN_ID] * batch_size,
                        input_token_id=args.input_token_id,
                    )
                )

        result["cases"].append(
            run_case(
                engine_core=engine_core,
                name="b4-independent-eos",
                batch_size=4,
                max_steps=4,
                expected_tokens=expected[4],
                eos_token_ids=[
                    expected[1][0],
                    DEFAULT_EOS_TOKEN_ID,
                    expected[1][0],
                    DEFAULT_EOS_TOKEN_ID,
                ],
                input_token_id=args.input_token_id,
            )
        )
        result["artifacts"] = {
            "baseline_result": str(baseline_result),
            "baseline_sha256": sha256(baseline_result),
            "native_library": os.environ.get(
                "VLLM_ASCEND_RESIDENT_EPOCH_LIBRARY"
            ),
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
        result["pass"] = (
            result["distributed_initialized_during_engine"]
            and all(case["pass"] for case in result["cases"])
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

    write_result(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
