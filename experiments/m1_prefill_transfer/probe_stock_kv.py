#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.request import Request

from run_scheduler_native import make_vllm_config


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def _tensor_record(value: Any) -> dict[str, Any]:
    return {
        "type": type(value).__name__,
        "shape": _shape(value),
        "dtype": str(getattr(value, "dtype", None)),
        "device": str(getattr(value, "device", None)),
        "contiguous": bool(value.is_contiguous())
        if isinstance(value, torch.Tensor)
        else None,
    }


def _cache_record(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_record(value)
    if isinstance(value, (tuple, list)):
        return [_cache_record(item) for item in value]
    return {"type": type(value).__name__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-token-ids",
        type=int,
        nargs="+",
        default=[9707, 11, 358],
    )
    args = parser.parse_args()

    config = make_vllm_config(args.model.resolve(strict=True))
    config.parallel_config.worker_cls = "vllm_ascend.worker.worker.NPUWorker"
    config.parallel_config.distributed_executor_backend = "uni"
    config.cache_config.gpu_memory_utilization = 0.35

    engine_core: EngineCore | None = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "prompt_token_ids": args.prompt_token_ids,
        "pass": False,
    }
    try:
        engine_core = EngineCore(
            vllm_config=config,
            executor_class=UniProcExecutor,
            log_stats=True,
        )
        params = SamplingParams(temperature=0.0, max_tokens=2)
        params.update_from_generation_config({}, 151645)
        request = Request(
            request_id="m1-kv-probe",
            client_index=0,
            prompt_token_ids=args.prompt_token_ids,
            sampling_params=params,
            pooling_params=None,
        )
        engine_core.add_request(request)
        outputs, model_executed = engine_core.step()
        engine_core.post_step(model_executed)

        worker = engine_core.model_executor.driver_worker.worker
        runner = worker.model_runner
        scheduler_request = engine_core.scheduler.requests[request.request_id]
        block_ids = engine_core.scheduler.kv_cache_manager.get_blocks(
            request.request_id
        ).get_block_ids()
        result.update(
            {
                "model_executed": model_executed,
                "output_client_ids": sorted(outputs),
                "worker": f"{type(worker).__module__}.{type(worker).__qualname__}",
                "runner": f"{type(runner).__module__}.{type(runner).__qualname__}",
                "num_computed_tokens": scheduler_request.num_computed_tokens,
                "num_tokens": scheduler_request.num_tokens,
                "num_output_tokens": scheduler_request.num_output_tokens,
                "scheduler_block_ids": block_ids,
                "kv_caches": [_cache_record(cache) for cache in runner.kv_caches],
            }
        )
        result["pass"] = bool(
            model_executed
            and len(runner.kv_caches) == 28
            and block_ids
            and scheduler_request.num_computed_tokens
            == len(args.prompt_token_ids)
        )
    finally:
        if engine_core is not None:
            engine_core.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
