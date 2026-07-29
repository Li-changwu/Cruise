#!/usr/bin/env python3
"""Run the M1 B=1-4 stock-prefill ownership-transfer differential gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import traceback
from typing import Any


SCHEDULER_QUALNAME = (
    "vllm_ascend_resident_epoch.scheduler.ResidentEpochScheduler"
)
WORKER_QUALNAME = "vllm_ascend.worker.worker.NPUWorker"
STOCK_SCHEDULER_QUALNAME = "vllm.v1.core.sched.scheduler.Scheduler"
GRAPH_BATCH_SIZE = 4
VOCAB_SIZE = 152064
IMPORT_INPUT_BYTES = 29_360_372
STEADY_INPUT_BYTES = 260
OUTPUT_BYTES = 368


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int

    def as_record(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class CaseSpec:
    name: str
    requests: tuple[RequestSpec, ...]

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "requests": [request.as_record() for request in self.requests],
        }


def load_case_manifest(path: Path) -> tuple[CaseSpec, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported batched-prefill case schema")
    logical_capacity = payload.get("logical_capacity")
    if logical_capacity != 8:
        raise ValueError("the current resident graph requires logical capacity 8")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case manifest must contain cases")

    cases: list[CaseSpec] = []
    case_names: set[str] = set()
    request_ids: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("each case must be an object")
        name = raw_case.get("name")
        if not isinstance(name, str) or not name or name in case_names:
            raise ValueError("case names must be non-empty and unique")
        case_names.add(name)
        raw_requests = raw_case.get("requests")
        if not isinstance(raw_requests, list) or not 1 <= len(raw_requests) <= 4:
            raise ValueError(f"{name}: request count must be between one and four")
        requests: list[RequestSpec] = []
        for raw_request in raw_requests:
            if not isinstance(raw_request, dict):
                raise ValueError(f"{name}: each request must be an object")
            request_id = raw_request.get("request_id")
            if (
                not isinstance(request_id, str)
                or not request_id
                or request_id in request_ids
            ):
                raise ValueError("request IDs must be non-empty and globally unique")
            request_ids.add(request_id)
            prompt = raw_request.get("prompt_token_ids")
            if not isinstance(prompt, list) or len(prompt) < 2:
                raise ValueError(
                    f"{request_id}: prompt must contain at least two tokens"
                )
            if any(
                not isinstance(token, int) or token < 0 or token >= VOCAB_SIZE
                for token in prompt
            ):
                raise ValueError(
                    f"{request_id}: prompt token is outside the vocabulary"
                )
            max_tokens = raw_request.get("max_tokens")
            if not isinstance(max_tokens, int) or max_tokens < 2:
                raise ValueError(f"{request_id}: max_tokens must be at least two")
            if len(prompt) + max_tokens - 1 > logical_capacity:
                raise ValueError(
                    f"{request_id}: prompt and output exceed resident capacity"
                )
            requests.append(
                RequestSpec(
                    request_id=request_id,
                    prompt_token_ids=tuple(prompt),
                    max_tokens=max_tokens,
                )
            )
        cases.append(CaseSpec(name=name, requests=tuple(requests)))

    if {case.batch_size for case in cases} != {1, 2, 3, 4}:
        raise ValueError("the M1 gate requires batch sizes 1, 2, 3, and 4")
    return tuple(cases)


def _plan_record(plan: Any) -> dict[str, Any] | None:
    from vllm_ascend_resident_epoch.contract import ResidentEpochPlan

    if not isinstance(plan, ResidentEpochPlan):
        return None
    return {
        "max_steps": plan.max_steps,
        "active_mask": list(plan.active_mask),
        "row_generations": list(plan.row_generations),
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
        "row_generations": list(result.row_generations),
        "feed_calls": result.feed_calls,
        "fetch_calls": result.fetch_calls,
        "declared_input_bytes": result.declared_input_bytes,
        "declared_output_bytes": result.declared_output_bytes,
        "kv_imported": result.kv_imported,
        "host_kv_checksum": result.kv_snapshot_checksum,
        "device_kv_checksum": result.kv_import_checksum,
    }


def _request_state(request: Any) -> dict[str, Any]:
    return {
        "status": request.status.name,
        "num_computed_tokens": request.num_computed_tokens,
        "num_output_tokens": request.num_output_tokens,
    }


def _tokens_by_request(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    tokens: dict[str, list[int]] = {}
    for record in records:
        tokens.setdefault(record["request_id"], []).extend(record["tokens"])
    return tokens


def _case_checks(
    spec: CaseSpec, case: dict[str, Any], mode: str
) -> dict[str, bool]:
    expected_ids = {request.request_id for request in spec.requests}
    expected_lengths = {
        request.request_id: request.max_tokens for request in spec.requests
    }
    steps = case["steps"]
    checks = {
        "request_drained": case["request_drained"] is True,
        "exact_request_set": set(case["tokens"]) == expected_ids,
        "exact_output_lengths": {
            req_id: len(tokens) for req_id, tokens in case["tokens"].items()
        }
        == expected_lengths,
        "all_requests_finished": all(
            state["status"].startswith("FINISHED_")
            for state in case["final_request_state"].values()
        ),
        "first_step_prefilled_all_requests": bool(
            steps
            and steps[0]["plan"] is None
            and set(steps[0]["new_tokens_by_request"]) == expected_ids
            and all(
                len(tokens) == 1
                for tokens in steps[0]["new_tokens_by_request"].values()
            )
        ),
        "cleanup_did_not_execute_model": bool(
            steps
            and steps[-1]["model_executed"] is False
            and not steps[-1]["engine_outputs"]
        ),
    }
    if mode == "baseline":
        checks.update(
            {
                "no_resident_plans": all(step["plan"] is None for step in steps),
                "no_resident_results": all(
                    step["result"] is None for step in steps
                ),
            }
        )
        return checks

    device_steps = [step for step in steps if step["plan"] is not None]
    import_steps = [
        step
        for step in device_steps
        if any(
            request["kv_import_required"] for request in step["plan"]["requests"]
        )
    ]
    import_step = import_steps[0] if len(import_steps) == 1 else None
    steady_steps = [step for step in device_steps if step is not import_step]

    stable_rows: dict[str, set[tuple[int, int]]] = {}
    for step in device_steps:
        for request in step["plan"]["requests"]:
            stable_rows.setdefault(request["req_id"], set()).add(
                (request["row"], request["generation"])
            )

    def valid_result(step: dict[str, Any], expected_input_bytes: int) -> bool:
        plan = step["plan"]
        result = step["result"]
        step_tokens = step["new_tokens_by_request"]
        return bool(
            result
            and result["route"] == "device"
            and result["status"] == 0
            and result["commit_state"] == "COMMITTED"
            and result["feed_calls"] == 1
            and result["fetch_calls"] == 1
            and result["declared_input_bytes"] == expected_input_bytes
            and result["declared_output_bytes"] == OUTPUT_BYTES
            and set(result["computed_steps"])
            == {request["req_id"] for request in plan["requests"]}
            and result["computed_steps"]
            == {req_id: len(tokens) for req_id, tokens in step_tokens.items()}
            and result["row_generations"] == plan["row_generations"]
        )

    import_valid = bool(
        import_step
        and {request["req_id"] for request in import_step["plan"]["requests"]}
        == expected_ids
        and import_step["plan"]["active_mask"]
        == [1] * spec.batch_size + [0] * (GRAPH_BATCH_SIZE - spec.batch_size)
        and all(
            request["state_owner"] == "host"
            and request["kv_import_required"]
            for request in import_step["plan"]["requests"]
        )
        and import_step["result"]
        and import_step["result"]["kv_imported"] is True
        and import_step["result"]["host_kv_checksum"] != 0
        and import_step["result"]["host_kv_checksum"]
        == import_step["result"]["device_kv_checksum"]
        and valid_result(import_step, IMPORT_INPUT_BYTES)
    )
    checks.update(
        {
            "only_prefill_used_host_execution": all(
                step["plan"] is not None
                for step in steps[1:]
                if step["model_executed"]
            ),
            "one_complete_batch_import": len(import_steps) == 1 and import_valid,
            "steady_epochs_device_owned": bool(steady_steps)
            and all(
                all(
                    request["state_owner"] == "device"
                    and not request["kv_import_required"]
                    for request in step["plan"]["requests"]
                )
                for step in steady_steps
            ),
            "one_feed_fetch_per_device_epoch": bool(device_steps)
            and all(
                valid_result(
                    step,
                    IMPORT_INPUT_BYTES if step is import_step else STEADY_INPUT_BYTES,
                )
                for step in device_steps
            ),
            "stable_row_generation": set(stable_rows) == expected_ids
            and all(len(values) == 1 for values in stable_rows.values()),
        }
    )
    if len(set(expected_lengths.values())) > 1:
        active_counts = [
            sum(step["plan"]["active_mask"]) for step in device_steps
        ]
        checks["mixed_completion_shrinks_batch"] = (
            bool(active_counts) and min(active_counts) < max(active_counts)
        )
    return checks


def _run_case(engine_core: Any, spec: CaseSpec, mode: str) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.v1.request import Request

    from run_scheduler_native import output_records

    requests: dict[str, Any] = {}
    for client_index, request_spec in enumerate(spec.requests):
        params = SamplingParams(temperature=0.0, max_tokens=request_spec.max_tokens)
        params.update_from_generation_config({}, 151645)
        request = Request(
            request_id=request_spec.request_id,
            client_index=client_index,
            prompt_token_ids=list(request_spec.prompt_token_ids),
            sampling_params=params,
            pooling_params=None,
        )
        requests[request_spec.request_id] = request
        engine_core.add_request(request)

    case: dict[str, Any] = {
        **spec.as_record(),
        "tokens": {request_id: [] for request_id in requests},
        "terminal_finish_reasons": {},
        "terminal_stop_reasons": {},
        "steps": [],
        "pass": False,
    }
    max_engine_steps = max(request.max_tokens for request in spec.requests) + 4
    for step_index in range(max_engine_steps):
        outputs, model_executed = engine_core.step()
        engine_core.post_step(model_executed)
        scheduler = engine_core.scheduler
        records = output_records(outputs)
        new_tokens = _tokens_by_request(records)
        unknown_ids = set(new_tokens) - set(requests)
        if unknown_ids:
            raise RuntimeError(f"{spec.name}: output contains unknown requests")
        for request_id, tokens in new_tokens.items():
            case["tokens"][request_id].extend(tokens)
        for record in records:
            if record["finish_reason"] is not None:
                case["terminal_finish_reasons"][record["request_id"]] = record[
                    "finish_reason"
                ]
                case["terminal_stop_reasons"][record["request_id"]] = record[
                    "stop_reason"
                ]
        case["steps"].append(
            {
                "index": step_index,
                "model_executed": model_executed,
                "new_tokens_by_request": new_tokens,
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

    case["request_drained"] = not engine_core.scheduler.has_requests()
    case["final_request_state"] = {
        request_id: _request_state(request)
        for request_id, request in requests.items()
    }
    case["checks"] = _case_checks(spec, case, mode)
    case["pass"] = all(case["checks"].values())
    return case


def run_engine(
    *, mode: str, model: Path, cases: tuple[CaseSpec, ...]
) -> dict[str, Any]:
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    from run_scheduler_native import make_vllm_config

    config = make_vllm_config(model)
    config.parallel_config.worker_cls = WORKER_QUALNAME
    config.parallel_config.distributed_executor_backend = "uni"
    config.cache_config.gpu_memory_utilization = 0.35
    if mode == "cruise":
        config.scheduler_config.scheduler_cls = SCHEDULER_QUALNAME

    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": "M1 simultaneous B=1-4 nontrivial stock-prefill transfer",
        "mode": mode,
        "model": str(model),
        "manifest": [case.as_record() for case in cases],
        "cases": [],
        "pass": False,
    }
    engine_core: Any | None = None
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
        for case in cases:
            result["cases"].append(_run_case(engine_core, case, mode))
        expected_scheduler = (
            STOCK_SCHEDULER_QUALNAME if mode == "baseline" else SCHEDULER_QUALNAME
        )
        result["checks"] = {
            "expected_scheduler": result["resolved_classes"]["scheduler"]
            == expected_scheduler,
            "all_batch_sizes_executed": {
                case["batch_size"] for case in result["cases"]
            }
            == {1, 2, 3, 4},
            "all_cases_passed": all(case["pass"] for case in result["cases"]),
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
    return result


def compare_results(baseline_path: Path, cruise_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    cruise = json.loads(cruise_path.read_text(encoding="utf-8"))
    baseline_cases = {case["name"]: case for case in baseline.get("cases", [])}
    cruise_cases = {case["name"]: case for case in cruise.get("cases", [])}
    case_names = set(baseline_cases) | set(cruise_cases)
    case_checks = {
        name: {
            "same_tokens": baseline_cases.get(name, {}).get("tokens")
            == cruise_cases.get(name, {}).get("tokens"),
            "same_finish_reasons": baseline_cases.get(name, {}).get(
                "terminal_finish_reasons"
            )
            == cruise_cases.get(name, {}).get("terminal_finish_reasons"),
            "same_stop_reasons": baseline_cases.get(name, {}).get(
                "terminal_stop_reasons"
            )
            == cruise_cases.get(name, {}).get("terminal_stop_reasons"),
            "same_final_request_state": baseline_cases.get(name, {}).get(
                "final_request_state"
            )
            == cruise_cases.get(name, {}).get("final_request_state"),
        }
        for name in sorted(case_names)
    }
    checks = {
        "baseline_passed": baseline.get("pass") is True,
        "cruise_passed": cruise.get("pass") is True,
        "same_manifest": baseline.get("manifest") == cruise.get("manifest"),
        "same_case_set": set(baseline_cases) == set(cruise_cases),
        "all_cases_equivalent": bool(case_checks)
        and all(all(values.values()) for values in case_checks.values()),
    }
    return {
        "schema_version": 1,
        "gate": "M1 B=1-4 batched prefill differential",
        "baseline": str(baseline_path),
        "cruise": str(cruise_path),
        "case_checks": case_checks,
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
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("cases.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cruise", type=Path)
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
        cases = load_case_manifest(args.cases.resolve(strict=True))
        result = run_engine(
            mode=args.mode,
            model=args.model.resolve(strict=True),
            cases=cases,
        )
    write_result(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
