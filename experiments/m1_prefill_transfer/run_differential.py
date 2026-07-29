#!/usr/bin/env python3
"""Run and compare stock-prefill and Cruise resident-decode executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any

SCHEDULER_QUALNAME = (
    "vllm_ascend_resident_epoch.scheduler.ResidentEpochScheduler"
)
WORKER_QUALNAME = "vllm_ascend.worker.worker.NPUWorker"


def _plan_record(plan: Any) -> dict[str, Any] | None:
    from vllm_ascend_resident_epoch.contract import ResidentEpochPlan

    if not isinstance(plan, ResidentEpochPlan):
        return None
    return {
        "max_steps": plan.max_steps,
        "active_mask": list(plan.active_mask),
        "requests": [
            {
                "req_id": request.req_id,
                "row": request.row,
                "generation": request.generation,
                "token_id": request.token_id,
                "position": request.position,
                "sequence_length": request.sequence_length,
                "scheduler_block_ids": list(request.scheduler_block_ids),
                "device_block_ids": list(request.device_block_ids),
                "state_owner": request.state_owner,
                "kv_import_required": request.kv_import_required,
            }
            for request in plan.requests
        ],
    }


def _result_record(result: Any) -> dict[str, Any] | None:
    from vllm_ascend_resident_epoch.contract import ResidentEpochResult

    if not isinstance(result, ResidentEpochResult):
        return None
    return {
        "route": result.route,
        "status": result.status,
        "commit_state": result.commit_state.name,
        "model_calls": result.model_calls,
        "computed_steps": result.computed_steps,
        "feed_calls": result.feed_calls,
        "fetch_calls": result.fetch_calls,
        "declared_input_bytes": result.declared_input_bytes,
        "declared_output_bytes": result.declared_output_bytes,
        "kv_imported": result.kv_imported,
        "host_kv_checksum": result.kv_snapshot_checksum,
        "device_kv_checksum": result.kv_import_checksum,
    }


def run_engine(
    *,
    mode: str,
    model: Path,
    prompt_token_ids: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.request import Request

    from run_scheduler_native import make_vllm_config, output_records

    config = make_vllm_config(model)
    config.parallel_config.worker_cls = WORKER_QUALNAME
    config.parallel_config.distributed_executor_backend = "uni"
    config.cache_config.gpu_memory_utilization = 0.35
    if mode == "cruise":
        config.scheduler_config.scheduler_cls = SCHEDULER_QUALNAME

    engine_core: EngineCore | None = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "model": str(model),
        "prompt_token_ids": prompt_token_ids,
        "max_tokens": max_tokens,
        "tokens": [],
        "steps": [],
        "pass": False,
    }
    try:
        engine_core = EngineCore(
            vllm_config=config,
            executor_class=UniProcExecutor,
            log_stats=True,
        )
        worker = engine_core.model_executor.driver_worker.worker
        result["resolved_classes"] = {
            "scheduler": (
                f"{type(engine_core.scheduler).__module__}."
                f"{type(engine_core.scheduler).__qualname__}"
            ),
            "worker": f"{type(worker).__module__}.{type(worker).__qualname__}",
            "runner": (
                f"{type(worker.model_runner).__module__}."
                f"{type(worker.model_runner).__qualname__}"
            ),
        }

        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        params.update_from_generation_config({}, 151645)
        request = Request(
            request_id=f"m1-{mode}",
            client_index=0,
            prompt_token_ids=prompt_token_ids,
            sampling_params=params,
            pooling_params=None,
        )
        engine_core.add_request(request)

        for step_index in range(max_tokens + 2):
            outputs, model_executed = engine_core.step()
            engine_core.post_step(model_executed)
            scheduler = engine_core.scheduler
            records = output_records(outputs)
            new_tokens = [
                token
                for record in records
                for token in record["tokens"]
                if record["request_id"] == request.request_id
            ]
            result["tokens"].extend(new_tokens)
            result["steps"].append(
                {
                    "index": step_index,
                    "model_executed": model_executed,
                    "new_tokens": new_tokens,
                    "engine_outputs": records,
                    "scheduler_rejection": getattr(
                        scheduler, "_resident_epoch_last_rejection", None
                    ),
                    "plan": _plan_record(
                        getattr(scheduler, "_resident_epoch_last_plan", None)
                    ),
                    "result": _result_record(
                        getattr(scheduler, "_resident_epoch_last_result", None)
                    ),
                }
            )
            if not scheduler.has_requests():
                break

        result["request_drained"] = not engine_core.scheduler.has_requests()
        result["checks"] = {
            "request_drained": result["request_drained"],
            "exact_output_length": len(result["tokens"]) == max_tokens,
            "token_steps_executed_model": all(
                step["model_executed"]
                for step in result["steps"]
                if step["new_tokens"]
            ),
            "cleanup_did_not_execute_model": bool(
                result["steps"]
                and not result["steps"][-1]["new_tokens"]
                and not result["steps"][-1]["model_executed"]
            ),
        }
        if mode == "baseline":
            model_steps = [
                step for step in result["steps"] if step["model_executed"]
            ]
            result["checks"].update(
                {
                    "stock_scheduler": result["resolved_classes"]["scheduler"]
                    == "vllm.v1.core.sched.scheduler.Scheduler",
                    "no_resident_plans": all(
                        step["plan"] is None for step in result["steps"]
                    ),
                    "one_output_per_host_step": len(model_steps) == max_tokens
                    and all(len(step["new_tokens"]) == 1 for step in model_steps),
                }
            )
        else:
            device_steps = [
                step for step in result["steps"] if step["plan"] is not None
            ]
            import_steps = [
                step
                for step in device_steps
                if step["plan"]["requests"][0]["kv_import_required"]
            ]
            steady_steps = [
                step
                for step in device_steps
                if step["plan"]["requests"][0]["state_owner"] == "device"
            ]
            checksum_equal = bool(
                len(import_steps) == 1
                and import_steps[0]["result"] is not None
                and import_steps[0]["result"]["host_kv_checksum"] != 0
                and import_steps[0]["result"]["host_kv_checksum"]
                == import_steps[0]["result"]["device_kv_checksum"]
            )
            result["checks"].update(
                {
                    "stock_prefill_first": bool(
                        result["steps"]
                        and result["steps"][0]["plan"] is None
                        and len(result["steps"][0]["new_tokens"]) == 1
                    ),
                    "one_import_epoch": len(import_steps) == 1,
                    "import_epoch_k2": bool(
                        import_steps and import_steps[0]["plan"]["max_steps"] == 2
                    ),
                    "host_device_kv_checksum_equal": checksum_equal,
                    "ownership_transferred": bool(steady_steps),
                    "steady_epoch_k1": bool(
                        steady_steps and steady_steps[0]["plan"]["max_steps"] == 1
                    ),
                    "one_feed_fetch_per_epoch": all(
                        step["result"] is not None
                        and step["result"]["feed_calls"] == 1
                        and step["result"]["fetch_calls"] == 1
                        for step in device_steps
                    ),
                    "steady_minimal_abi": bool(
                        steady_steps
                        and steady_steps[0]["result"] is not None
                        and steady_steps[0]["result"]["declared_input_bytes"] == 260
                        and steady_steps[0]["result"]["declared_output_bytes"] == 368
                    ),
                }
            )
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
    return result


def compare_results(baseline_path: Path, cruise_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    cruise = json.loads(cruise_path.read_text(encoding="utf-8"))
    checks = {
        "baseline_passed": baseline.get("pass") is True,
        "cruise_passed": cruise.get("pass") is True,
        "same_prompt": baseline.get("prompt_token_ids")
        == cruise.get("prompt_token_ids"),
        "same_output_tokens": baseline.get("tokens") == cruise.get("tokens"),
    }
    return {
        "schema_version": 1,
        "gate": "M1 stock-prefill to device-resident paged-KV ownership transfer",
        "baseline": str(baseline_path),
        "cruise": str(cruise_path),
        "baseline_tokens": baseline.get("tokens"),
        "cruise_tokens": cruise.get("tokens"),
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
    parser.add_argument(
        "--mode", choices=("baseline", "cruise", "compare"), required=True
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cruise", type=Path)
    parser.add_argument(
        "--prompt-token-ids", type=int, nargs="+", default=[9707, 11, 358]
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()

    if args.mode == "compare":
        if args.baseline is None or args.cruise is None:
            parser.error("compare mode requires --baseline and --cruise")
        result = compare_results(
            args.baseline.resolve(strict=True), args.cruise.resolve(strict=True)
        )
    else:
        if args.model is None:
            parser.error(f"{args.mode} mode requires --model")
        result = run_engine(
            mode=args.mode,
            model=args.model.resolve(strict=True),
            prompt_token_ids=args.prompt_token_ids,
            max_tokens=args.max_tokens,
        )
    write_result(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
