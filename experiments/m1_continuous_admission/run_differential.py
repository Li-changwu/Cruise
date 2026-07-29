#!/usr/bin/env python3
"""Run the M1 continuous-admission and row-reuse differential gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any

from experiments.m1_batched_prefill.run_differential import (
    IMPORT_INPUT_BYTES,
    OUTPUT_BYTES,
    SCHEDULER_QUALNAME,
    STEADY_INPUT_BYTES,
    STOCK_SCHEDULER_QUALNAME,
    WORKER_QUALNAME,
    RequestSpec,
    _plan_record,
    _request_state,
    _result_record,
    _tokens_by_request,
    write_result,
)


def load_scenario(path: Path) -> tuple[RequestSpec, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported continuous-admission scenario schema")
    if payload.get("logical_capacity") != 8:
        raise ValueError("the current resident graph requires logical capacity 8")
    if payload.get("initial_output_boundary") != 3:
        raise ValueError("the continuous-admission gate requires A:3")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list) or len(raw_requests) != 3:
        raise ValueError("the continuous-admission gate requires A, B, and C")

    requests: list[RequestSpec] = []
    expected_ids = ("A", "B", "C")
    expected_admission = (None, "A:3", "B:complete")
    for raw, expected_id, admit_after in zip(
        raw_requests, expected_ids, expected_admission, strict=True
    ):
        if not isinstance(raw, dict) or raw.get("request_id") != expected_id:
            raise ValueError("scenario requests must be ordered A, B, C")
        if raw.get("admit_after") != admit_after:
            raise ValueError(f"{expected_id}: invalid admission boundary")
        prompt = raw.get("prompt_token_ids")
        max_tokens = raw.get("max_tokens")
        if not isinstance(prompt, list) or len(prompt) < 2:
            raise ValueError(f"{expected_id}: prompt must be nontrivial")
        if not isinstance(max_tokens, int) or max_tokens < 2:
            raise ValueError(f"{expected_id}: output budget must be at least two")
        if len(prompt) + max_tokens - 1 > 8:
            raise ValueError(f"{expected_id}: request exceeds resident capacity")
        requests.append(
            RequestSpec(
                request_id=expected_id,
                prompt_token_ids=tuple(prompt),
                max_tokens=max_tokens,
            )
        )
    return tuple(requests)


def _terminal_records(records: list[dict[str, Any]], result: dict[str, Any]) -> None:
    for record in records:
        if record["finish_reason"] is None:
            continue
        result["terminal_finish_reasons"][record["request_id"]] = record[
            "finish_reason"
        ]
        result["terminal_stop_reasons"][record["request_id"]] = record[
            "stop_reason"
        ]


def _device_step_valid(step: dict[str, Any], importing: bool) -> bool:
    plan = step["plan"]
    result = step["result"]
    expected_input = IMPORT_INPUT_BYTES if importing else STEADY_INPUT_BYTES
    return bool(
        plan
        and result
        and result["route"] == "device"
        and result["status"] == 0
        and result["commit_state"] == "COMMITTED"
        and result["feed_calls"] == 1
        and result["fetch_calls"] == 1
        and result["declared_input_bytes"] == expected_input
        and result["declared_output_bytes"] == OUTPUT_BYTES
        and result["computed_steps"]
        == {
            req_id: len(tokens)
            for req_id, tokens in step["new_tokens_by_request"].items()
        }
        and result["row_generations"] == plan["row_generations"]
        and (
            not importing
            or (
                result["kv_imported"] is True
                and result["host_kv_checksum"] != 0
                and result["host_kv_checksum"]
                == result["device_kv_checksum"]
            )
        )
    )


def _checks(
    mode: str, specs: tuple[RequestSpec, ...], result: dict[str, Any]
) -> dict[str, bool]:
    expected_ids = {request.request_id for request in specs}
    expected_lengths = {
        request.request_id: request.max_tokens for request in specs
    }
    checks = {
        "request_drained": result["request_drained"] is True,
        "exact_output_lengths": {
            req_id: len(tokens) for req_id, tokens in result["tokens"].items()
        }
        == expected_lengths,
        "terminal_reason_per_request": set(result["terminal_finish_reasons"])
        == expected_ids,
        "final_state_per_request": set(result["final_request_state"])
        == expected_ids
        and all(
            state["status"].startswith("FINISHED_")
            for state in result["final_request_state"].values()
        ),
        "cleanup_did_not_execute_model": bool(
            result["steps"]
            and result["steps"][-1]["model_executed"] is False
            and not result["steps"][-1]["engine_outputs"]
        ),
    }
    if mode == "baseline":
        checks["stock_only"] = all(
            step["plan"] is None and step["result"] is None
            for step in result["steps"]
        )
        return checks

    model_steps = [step for step in result["steps"] if step["model_executed"]]
    host_steps = [step for step in model_steps if step["plan"] is None]
    device_steps = [step for step in model_steps if step["plan"] is not None]
    expected_host_sets = [{"A"}, {"B"}, {"C"}]
    actual_host_sets = [set(step["new_tokens_by_request"]) for step in host_steps]
    expected_device_sets = [{"A"}, {"A", "B"}, {"A", "C"}, {"A"}]
    actual_device_sets = [
        {request["req_id"] for request in step["plan"]["requests"]}
        for step in device_steps
    ]
    imported_ids = [
        {
            request["req_id"]
            for request in step["plan"]["requests"]
            if request["kv_import_required"]
        }
        for step in device_steps
    ]
    rows = {
        request["req_id"]: (request["row"], request["generation"])
        for step in device_steps
        for request in step["plan"]["requests"]
        if request["req_id"] in {"B", "C"}
    }
    a_rows = {
        (request["row"], request["generation"])
        for step in device_steps
        for request in step["plan"]["requests"]
        if request["req_id"] == "A"
    }
    checks.update(
        {
            "isolated_host_prefills": actual_host_sets == expected_host_sets
            and [step["scheduler_rejection"] for step in host_steps]
            == [
                "not-single-token-decode",
                "host-prefill-admission",
                "host-prefill-admission",
            ],
            "device_epoch_request_sets": actual_device_sets
            == expected_device_sets,
            "epoch_steps": [step["plan"]["max_steps"] for step in device_steps]
            == [2, 1, 1, 2],
            "selective_imports": imported_ids
            == [{"A"}, {"B"}, {"C"}, set()],
            "row_reused_with_new_generation": rows
            == {"B": (1, 2), "C": (1, 3)},
            "device_owned_a_stable": a_rows == {(0, 1)}
            and all(
                next(
                    request
                    for request in step["plan"]["requests"]
                    if request["req_id"] == "A"
                )["state_owner"]
                == ("host" if index == 0 else "device")
                for index, step in enumerate(device_steps)
            ),
            "device_results_valid": len(device_steps) == 4
            and all(
                _device_step_valid(step, bool(imported_ids[index]))
                for index, step in enumerate(device_steps)
            ),
        }
    )
    return checks


def run_engine(
    *, mode: str, model: Path, specs: tuple[RequestSpec, ...]
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

    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": "M1 continuous nontrivial-prefill admission and row reuse",
        "mode": mode,
        "model": str(model),
        "scenario": [request.as_record() for request in specs],
        "tokens": {request.request_id: [] for request in specs},
        "terminal_finish_reasons": {},
        "terminal_stop_reasons": {},
        "steps": [],
        "pass": False,
    }
    engine_core: Any | None = None
    requests: dict[str, Any] = {}

    def add_request(spec: RequestSpec) -> None:
        params = SamplingParams(temperature=0.0, max_tokens=spec.max_tokens)
        params.update_from_generation_config({}, 151645)
        request = Request(
            request_id=spec.request_id,
            client_index=ord(spec.request_id) - ord("A"),
            prompt_token_ids=list(spec.prompt_token_ids),
            sampling_params=params,
            pooling_params=None,
        )
        requests[spec.request_id] = request
        engine_core.add_request(request)

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
        }
        add_request(specs[0])
        admitted = {"A"}
        for step_index in range(16):
            outputs, model_executed = engine_core.step()
            engine_core.post_step(model_executed)
            scheduler = engine_core.scheduler
            records = output_records(outputs)
            new_tokens = _tokens_by_request(records)
            for request_id, tokens in new_tokens.items():
                result["tokens"][request_id].extend(tokens)
            _terminal_records(records, result)
            result["steps"].append(
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
            if "B" not in admitted and len(result["tokens"]["A"]) >= 3:
                add_request(specs[1])
                admitted.add("B")
                result["steps"][-1]["admitted_after_step"] = "B"
            elif (
                "B" in admitted
                and "C" not in admitted
                and len(result["tokens"]["B"]) == specs[1].max_tokens
            ):
                add_request(specs[2])
                admitted.add("C")
                result["steps"][-1]["admitted_after_step"] = "C"
            if admitted == {"A", "B", "C"} and not scheduler.has_requests():
                break

        result["request_drained"] = not engine_core.scheduler.has_requests()
        result["final_request_state"] = {
            request_id: _request_state(request)
            for request_id, request in requests.items()
        }
        expected_scheduler = (
            STOCK_SCHEDULER_QUALNAME if mode == "baseline" else SCHEDULER_QUALNAME
        )
        result["checks"] = _checks(mode, specs, result)
        result["checks"]["expected_scheduler"] = (
            result["resolved_classes"]["scheduler"] == expected_scheduler
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
        "same_scenario": baseline.get("scenario") == cruise.get("scenario"),
        "same_tokens": baseline.get("tokens") == cruise.get("tokens"),
        "same_finish_reasons": baseline.get("terminal_finish_reasons")
        == cruise.get("terminal_finish_reasons"),
        "same_stop_reasons": baseline.get("terminal_stop_reasons")
        == cruise.get("terminal_stop_reasons"),
        "same_final_request_state": baseline.get("final_request_state")
        == cruise.get("final_request_state"),
    }
    return {
        "schema_version": 1,
        "gate": "M1 continuous-admission differential",
        "baseline": str(baseline_path),
        "cruise": str(cruise_path),
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("baseline", "cruise", "compare"), required=True
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("scenario.json"),
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
        result = run_engine(
            mode=args.mode,
            model=args.model.resolve(strict=True),
            specs=load_scenario(args.cases.resolve(strict=True)),
        )
    write_result(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
