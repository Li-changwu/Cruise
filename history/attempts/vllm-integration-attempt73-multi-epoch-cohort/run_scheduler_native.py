#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_ascend_resident_epoch.config import ResidentEpochConfig
from vllm_ascend_resident_epoch.contract import get_plan, get_result
from vllm_ascend_resident_epoch.scheduler import ResidentEpochScheduler
from vllm_ascend_resident_epoch.worker import ResidentEpochWorker


DEFAULT_EOS_TOKEN_ID = 151645
DEFAULT_INPUT_TOKEN_ID = 11690
GRAPH_BATCH_SIZE = 4
BLOCK_SIZE = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_tokens(path: Path) -> dict[int, list[int]]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("pass") is not True:
        raise RuntimeError("the frozen Attempt 69e result is not passing evidence")

    expected: dict[int, list[int]] = {}
    for max_steps in (1, 2, 4, 8):
        prefix = f"k{max_steps}-"
        matches = [
            case
            for case in evidence["cases"]
            if case["case"].startswith(prefix)
            and case["max_steps"] == max_steps
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one frozen baseline case for K={max_steps}")
        steps = sorted(
            (
                item
                for item in matches[0]["steps"]
                if item["request"] == 0
            ),
            key=lambda item: item["step"],
        )
        tokens = [int(item["device_token"]) for item in steps]
        if len(tokens) != max_steps:
            raise RuntimeError(f"incomplete frozen token sequence for K={max_steps}")
        expected[max_steps] = tokens
    return expected


def make_vllm_config(model: Path) -> VllmConfig:
    model_config = ModelConfig(
        model=str(model),
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=BLOCK_SIZE,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=GRAPH_BATCH_SIZE,
        max_num_batched_tokens=BLOCK_SIZE,
        max_model_len=BLOCK_SIZE,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    cache_config.num_gpu_blocks = 8
    return VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )


def make_scheduler(vllm_config: VllmConfig) -> ResidentEpochScheduler:
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.bfloat16,
                ),
            )
        ],
    )
    return ResidentEpochScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def make_request(
    request_id: str,
    client_index: int,
    max_tokens: int,
    eos_token_id: int,
    input_token_id: int,
) -> Request:
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    params.update_from_generation_config({}, eos_token_id)
    return Request(
        request_id=request_id,
        client_index=client_index,
        prompt_token_ids=[input_token_id],
        sampling_params=params,
        pooling_params=None,
    )


def output_records(outputs: dict[int, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for client_index in sorted(outputs):
        for output in outputs[client_index].outputs:
            finish_reason = output.finish_reason
            records.append(
                {
                    "client_index": client_index,
                    "request_id": output.request_id,
                    "tokens": list(output.new_token_ids),
                    "finish_reason": (
                        finish_reason.value
                        if hasattr(finish_reason, "value")
                        else finish_reason
                    ),
                    "stop_reason": output.stop_reason,
                }
            )
    return records


def run_case(
    worker: ResidentEpochWorker,
    vllm_config: VllmConfig,
    name: str,
    batch_size: int,
    max_steps: int,
    expected_tokens: list[int],
    eos_token_ids: list[int],
    input_token_id: int,
) -> dict[str, Any]:
    scheduler = make_scheduler(vllm_config)
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
        scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    plan = get_plan(scheduler_output)
    if plan is None:
        raise RuntimeError(
            f"{name}: scheduler rejected native plan: "
            f"{scheduler._resident_epoch_last_rejection}"
        )

    execute_start = time.perf_counter_ns()
    model_output = worker.execute_model(scheduler_output)
    execute_wall_us = (time.perf_counter_ns() - execute_start) // 1000
    result = get_result(model_output)
    if result is None:
        raise RuntimeError(f"{name}: native output has no execution metadata")
    engine_outputs = scheduler.update_from_output(scheduler_output, model_output)

    sampled = {
        request_id: list(
            model_output.sampled_token_ids[
                model_output.req_id_to_index[request_id]
            ]
        )
        for request_id in plan.req_ids
    }
    expected_by_request = {
        request.request_id: (
            expected_tokens[:1]
            if eos_token_ids[row] == expected_tokens[0]
            else expected_tokens
        )
        for row, request in enumerate(requests)
    }
    checks = {
        "fixed_b4_plan": plan.graph_batch_size == GRAPH_BATCH_SIZE,
        "dense_active_mask": plan.active_mask
        == (1,) * batch_size + (0,) * (GRAPH_BATCH_SIZE - batch_size),
        "single_scheduler_token": all(
            count == 1 for count in scheduler_output.num_scheduled_tokens.values()
        ),
        "scheduler_kv_blocks_present": all(
            len(request.scheduler_block_ids) == 1 for request in plan.requests
        ),
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
    }
    return {
        "name": name,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "input_token_id": input_token_id,
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
        "worker_execute_wall_us": execute_wall_us,
        "request_status": {
            request.request_id: request.status.name for request in requests
        },
        "engine_outputs": output_records(engine_outputs),
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
        "gate": "Attempt 71 real vLLM SchedulerOutput to DataFlow native epoch",
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
    worker: ResidentEpochWorker | None = None
    try:
        model_config = args.model_config.resolve(strict=True)
        baseline_result = args.baseline_result.resolve(strict=True)
        expected = load_expected_tokens(baseline_result)
        vllm_config = make_vllm_config(model_config)
        worker = ResidentEpochWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
        )
        worker.init_device()
        load_start = time.perf_counter_ns()
        worker.load_model()
        result["native_load_wall_us"] = (
            time.perf_counter_ns() - load_start
        ) // 1000

        for batch_size in (1, 2, 4):
            for max_steps in (1, 2, 4, 8):
                result["cases"].append(
                    run_case(
                        worker=worker,
                        vllm_config=vllm_config,
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
                worker=worker,
                vllm_config=vllm_config,
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
            "native_server": os.environ.get(
                "VLLM_ASCEND_RESIDENT_EPOCH_SERVER"
            ),
            "backend_factory": os.environ.get(
                "VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY"
            ),
            "air": os.environ.get("VLLM_ASCEND_RESIDENT_EPOCH_AIR"),
            "tiling": os.environ.get("VLLM_ASCEND_RESIDENT_EPOCH_TILING"),
            "external_weights": os.environ.get(
                "VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS"
            ),
        }
        result["pass"] = all(case["pass"] for case in result["cases"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if worker is not None:
            try:
                worker.shutdown()
            except Exception as exc:
                result["shutdown_error"] = f"{type(exc).__name__}: {exc}"
                result["pass"] = False

    write_result(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
