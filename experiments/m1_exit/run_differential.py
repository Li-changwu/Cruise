#!/usr/bin/env python3
"""Run the 1,000-request M1 EngineCore differential exit gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

from experiments.m1_batched_prefill.run_differential import (
    SCHEDULER_QUALNAME,
    STOCK_SCHEDULER_QUALNAME,
    WORKER_QUALNAME,
    configured_kv_cache_bytes,
    _plan_record,
    _request_state,
    _result_record,
    _tokens_by_request,
    write_result,
)


DEFAULT_EOS_TOKEN_ID = 151645
REQUEST_KINDS = (
    "eligible",
    "unsupported_min_tokens",
    "eos_second_token",
    "cancel_after_prefill",
    "cancel_after_device",
)


@dataclass(frozen=True)
class PromptTemplate:
    token_ids: tuple[int, ...]
    second_output_token_id: int


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    kind: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    eos_token_id: int
    cancel_after: int | None

    def as_record(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "kind": self.kind,
            "prompt_token_ids": list(self.prompt_token_ids),
            "max_tokens": self.max_tokens,
            "eos_token_id": self.eos_token_id,
            "cancel_after": self.cancel_after,
        }


@dataclass(frozen=True)
class CohortSpec:
    name: str
    batch_size: int
    requests: tuple[RequestSpec, ...]


@dataclass(frozen=True)
class Workload:
    logical_capacity: int
    cohorts: tuple[CohortSpec, ...]

    @property
    def requests(self) -> tuple[RequestSpec, ...]:
        return tuple(request for cohort in self.cohorts for request in cohort.requests)

    def summary(self) -> dict[str, Any]:
        kind_counts = Counter(request.kind for request in self.requests)
        prompt_lengths = Counter(
            len(request.prompt_token_ids) for request in self.requests
        )
        output_budgets = Counter(request.max_tokens for request in self.requests)
        return {
            "cohort_count": len(self.cohorts),
            "request_count": len(self.requests),
            "batch_size_counts": dict(
                sorted(Counter(cohort.batch_size for cohort in self.cohorts).items())
            ),
            "kind_counts": dict(sorted(kind_counts.items())),
            "prompt_length_counts": dict(sorted(prompt_lengths.items())),
            "output_budget_counts": dict(sorted(output_budgets.items())),
        }

    def digest(self) -> str:
        records = [request.as_record() for request in self.requests]
        payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request_kind(batch_size: int, cohort_index: int, row: int) -> str:
    selector = (cohort_index + row + batch_size * 3) % 20
    if selector == 0:
        return "eos_second_token"
    if selector == 1:
        return "cancel_after_prefill"
    if selector == 2:
        return "cancel_after_device"
    if selector == 3:
        return "unsupported_min_tokens"
    return "eligible"


def load_workload(path: Path) -> Workload:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported M1 exit workload schema")
    logical_capacity = payload.get("logical_capacity")
    if logical_capacity != 8:
        raise ValueError("the current resident graph requires logical capacity 8")
    cohorts_per_batch = payload.get("cohorts_per_batch_size")
    if cohorts_per_batch != 100:
        raise ValueError("the M1 exit gate requires 100 cohorts per batch size")

    raw_templates = payload.get("prompt_templates")
    if not isinstance(raw_templates, list) or len(raw_templates) != 4:
        raise ValueError("the M1 exit gate requires four prompt templates")
    templates: list[PromptTemplate] = []
    for expected_length, raw in enumerate(raw_templates, start=2):
        tokens = raw.get("prompt_token_ids") if isinstance(raw, dict) else None
        second_token = (
            raw.get("second_output_token_id") if isinstance(raw, dict) else None
        )
        if (
            not isinstance(tokens, list)
            or len(tokens) != expected_length
            or not all(isinstance(token, int) and token >= 0 for token in tokens)
            or not isinstance(second_token, int)
            or second_token < 0
        ):
            raise ValueError("invalid M1 exit prompt template")
        templates.append(PromptTemplate(tuple(tokens), second_token))

    cohorts: list[CohortSpec] = []
    for batch_size in range(1, 5):
        for cohort_index in range(cohorts_per_batch):
            requests: list[RequestSpec] = []
            for row in range(batch_size):
                template = templates[(cohort_index + row + batch_size) % len(templates)]
                kind = _request_kind(batch_size, cohort_index, row)
                max_allowed = logical_capacity - len(template.token_ids) + 1
                max_tokens = 2 + (cohort_index + row) % (max_allowed - 1)
                if kind == "eos_second_token":
                    max_tokens = max(3, max_tokens)
                if kind == "cancel_after_device":
                    max_tokens = max(4, max_tokens)
                cancel_after = {
                    "cancel_after_prefill": 1,
                    "cancel_after_device": 3,
                }.get(kind)
                eos_token_id = (
                    template.second_output_token_id
                    if kind == "eos_second_token"
                    else DEFAULT_EOS_TOKEN_ID
                )
                request_id = f"b{batch_size}-c{cohort_index:03d}-r{row}"
                requests.append(
                    RequestSpec(
                        request_id=request_id,
                        kind=kind,
                        prompt_token_ids=template.token_ids,
                        max_tokens=max_tokens,
                        eos_token_id=eos_token_id,
                        cancel_after=cancel_after,
                    )
                )
            cohorts.append(
                CohortSpec(
                    name=f"b{batch_size}-c{cohort_index:03d}",
                    batch_size=batch_size,
                    requests=tuple(requests),
                )
            )

    workload = Workload(logical_capacity=logical_capacity, cohorts=tuple(cohorts))
    summary = workload.summary()
    if summary["request_count"] != 1000:
        raise ValueError("the M1 exit workload must contain exactly 1,000 requests")
    if set(summary["batch_size_counts"]) != {1, 2, 3, 4}:
        raise ValueError("the M1 exit workload must cover batch sizes 1-4")
    if set(summary["kind_counts"]) != set(REQUEST_KINDS):
        raise ValueError("the M1 exit workload must cover every request kind")
    return workload


def _expected_request(spec: RequestSpec) -> tuple[int, str]:
    if spec.kind == "eos_second_token":
        return 2, "FINISHED_STOPPED"
    if spec.cancel_after is not None:
        return spec.cancel_after, "FINISHED_ABORTED"
    return spec.max_tokens, "FINISHED_LENGTH_CAPPED"


def _run_cohort(
    engine_core: Any,
    spec: CohortSpec,
    mode: str,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.request import Request

    from run_scheduler_native import output_records

    requests: dict[str, Any] = {}
    specs = {request.request_id: request for request in spec.requests}
    for client_index, request_spec in enumerate(spec.requests):
        params = SamplingParams(
            temperature=0.0,
            min_tokens=1 if request_spec.kind == "unsupported_min_tokens" else 0,
            max_tokens=request_spec.max_tokens,
        )
        params.update_from_generation_config({}, request_spec.eos_token_id)
        if request_spec.cancel_after is not None:
            params.output_kind = RequestOutputKind.DELTA
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
        "name": spec.name,
        "batch_size": spec.batch_size,
        "tokens": {request_id: [] for request_id in requests},
        "terminal_finish_reasons": {},
        "terminal_stop_reasons": {},
        "cancelled_at": {},
        "pass": False,
    }
    failure_trace: list[dict[str, Any]] = []
    max_engine_steps = max(request.max_tokens for request in spec.requests) + 8
    for step_index in range(max_engine_steps):
        scheduler = engine_core.scheduler
        device_owned_before = set(
            getattr(scheduler, "_resident_epoch_device_owned", set())
        )
        outputs, model_executed = engine_core.step()
        engine_core.post_step(model_executed)
        records = output_records(outputs)
        new_tokens = _tokens_by_request(records)
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

        plan = getattr(scheduler, "_resident_epoch_last_plan", None)
        result = getattr(scheduler, "_resident_epoch_last_result", None)
        plan_record = _plan_record(plan)
        result_record = _result_record(result)
        route = "device" if plan_record is not None else "host"
        if model_executed:
            aggregate[f"{route}_model_steps"] += 1
            for request_id in new_tokens:
                aggregate[f"{route}_request_steps"][request_id] += 1
        if route == "host":
            aggregate["host_device_owned_violations"].update(
                device_owned_before & set(new_tokens)
            )
        else:
            assert plan_record is not None
            plan_ids = {request["req_id"] for request in plan_record["requests"]}
            aggregate["ineligible_device_requests"].update(
                request_id
                for request_id in plan_ids
                if specs[request_id].kind == "unsupported_min_tokens"
            )
            if result_record is None:
                aggregate["invalid_device_results"] += 1
            else:
                if (
                    result_record["feed_calls"] != 1
                    or result_record["fetch_calls"] != 1
                    or result_record["status"] != 0
                    or result_record["commit_state"] != "COMMITTED"
                ):
                    aggregate["invalid_device_results"] += 1
                if result_record["kv_imported"]:
                    aggregate["kv_imports"] += 1
                    if (
                        result_record["host_kv_checksum"] == 0
                        or result_record["host_kv_checksum"]
                        != result_record["device_kv_checksum"]
                    ):
                        aggregate["kv_checksum_mismatches"] += 1
                if any(
                    specs[request_id].kind == "eos_second_token"
                    and result_record["computed_steps"].get(request_id, 0)
                    < plan_record["max_steps"]
                    for request_id in plan_ids
                ):
                    aggregate["short_eos_device_epochs"] += 1
            for request in plan_record["requests"]:
                request_id = request["req_id"]
                lease = (request["row"], request["generation"])
                aggregate["leases"][request_id].add(lease)
                owner = aggregate["lease_owners"].setdefault(lease, request_id)
                if owner != request_id:
                    aggregate["lease_aliases"] += 1
                previous = aggregate["last_row_lease"].get(request["row"])
                if previous is not None and previous[0] != request_id:
                    if request["generation"] <= previous[1]:
                        aggregate["nonmonotonic_row_reuse"] += 1
                    else:
                        aggregate["row_reuses"] += 1
                aggregate["last_row_lease"][request["row"]] = (
                    request_id,
                    request["generation"],
                )

        cancel_ids: list[str] = []
        for request_id, request_spec in specs.items():
            if (
                request_spec.cancel_after is not None
                and request_id not in case["cancelled_at"]
                and len(case["tokens"][request_id]) == request_spec.cancel_after
                and not requests[request_id].status.name.startswith("FINISHED_")
            ):
                cancel_ids.append(request_id)
        if cancel_ids:
            owned_at_cancel = device_owned_before | set(
                getattr(scheduler, "_resident_epoch_device_owned", set())
            )
            aggregate["device_owned_cancellations"] += len(
                set(cancel_ids) & owned_at_cancel
            )
            engine_core.abort_requests(cancel_ids)
            for request_id in cancel_ids:
                case["cancelled_at"][request_id] = len(case["tokens"][request_id])

        if model_executed and (
            set(new_tokens) - set(requests)
            or (route == "host" and device_owned_before & set(new_tokens))
        ):
            failure_trace.append(
                {
                    "step": step_index,
                    "route": route,
                    "new_tokens": new_tokens,
                    "device_owned_before": sorted(device_owned_before),
                    "plan": plan_record,
                    "result": result_record,
                }
            )
        if not scheduler.has_requests():
            break

    case["request_drained"] = not engine_core.scheduler.has_requests()
    case["final_request_state"] = {
        request_id: _request_state(request)
        for request_id, request in requests.items()
    }
    expected_lengths = {
        request.request_id: _expected_request(request)[0]
        for request in spec.requests
    }
    expected_statuses = {
        request.request_id: _expected_request(request)[1]
        for request in spec.requests
    }
    checks = {
        "request_drained": case["request_drained"],
        "exact_request_set": set(case["tokens"]) == set(requests),
        "exact_output_lengths": {
            request_id: len(tokens) for request_id, tokens in case["tokens"].items()
        }
        == expected_lengths,
        "expected_statuses": {
            request_id: state["status"]
            for request_id, state in case["final_request_state"].items()
        }
        == expected_statuses,
        "expected_cancellations": case["cancelled_at"]
        == {
            request.request_id: request.cancel_after
            for request in spec.requests
            if request.cancel_after is not None
        },
        "eos_tokens_observed": all(
            request.kind != "eos_second_token"
            or case["tokens"][request.request_id][-1] == request.eos_token_id
            for request in spec.requests
        ),
    }
    if mode == "baseline":
        checks["stock_only"] = aggregate["device_model_steps"] == 0
    case["checks"] = checks
    case["pass"] = all(checks.values())
    if failure_trace:
        case["failure_trace"] = failure_trace
    return case


def _new_aggregate() -> dict[str, Any]:
    return {
        "host_model_steps": 0,
        "device_model_steps": 0,
        "host_request_steps": defaultdict(int),
        "device_request_steps": defaultdict(int),
        "host_device_owned_violations": set(),
        "ineligible_device_requests": set(),
        "invalid_device_results": 0,
        "kv_imports": 0,
        "kv_checksum_mismatches": 0,
        "short_eos_device_epochs": 0,
        "device_owned_cancellations": 0,
        "leases": defaultdict(set),
        "lease_owners": {},
        "lease_aliases": 0,
        "last_row_lease": {},
        "row_reuses": 0,
        "nonmonotonic_row_reuse": 0,
    }


def _aggregate_record(aggregate: dict[str, Any]) -> dict[str, Any]:
    stable_leases = all(len(values) == 1 for values in aggregate["leases"].values())
    return {
        "host_model_steps": aggregate["host_model_steps"],
        "device_model_steps": aggregate["device_model_steps"],
        "host_routed_request_count": len(aggregate["host_request_steps"]),
        "device_routed_request_count": len(aggregate["device_request_steps"]),
        "host_device_owned_violations": sorted(
            aggregate["host_device_owned_violations"]
        ),
        "ineligible_device_requests": sorted(
            aggregate["ineligible_device_requests"]
        ),
        "invalid_device_results": aggregate["invalid_device_results"],
        "kv_imports": aggregate["kv_imports"],
        "kv_checksum_mismatches": aggregate["kv_checksum_mismatches"],
        "short_eos_device_epochs": aggregate["short_eos_device_epochs"],
        "device_owned_cancellations": aggregate["device_owned_cancellations"],
        "stable_request_leases": stable_leases,
        "lease_aliases": aggregate["lease_aliases"],
        "row_reuses": aggregate["row_reuses"],
        "nonmonotonic_row_reuse": aggregate["nonmonotonic_row_reuse"],
    }


def run_engine(*, mode: str, model: Path, workload: Workload) -> dict[str, Any]:
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    from run_scheduler_native import make_vllm_config

    config = make_vllm_config(model)
    config.parallel_config.worker_cls = WORKER_QUALNAME
    config.parallel_config.distributed_executor_backend = "uni"
    config.cache_config.gpu_memory_utilization = 0.35
    config.cache_config.kv_cache_memory_bytes = configured_kv_cache_bytes()
    if mode == "cruise":
        config.scheduler_config.scheduler_cls = SCHEDULER_QUALNAME

    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": "M1 1,000-request EngineCore differential",
        "mode": mode,
        "model": str(model),
        "workload_sha256": workload.digest(),
        "workload_summary": workload.summary(),
        "cases": [],
        "pass": False,
    }
    aggregate = _new_aggregate()
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
        }
        for cohort in workload.cohorts:
            result["cases"].append(
                _run_cohort(engine_core, cohort, mode, aggregate)
            )
        result["aggregate"] = _aggregate_record(aggregate)
        expected_scheduler = (
            STOCK_SCHEDULER_QUALNAME if mode == "baseline" else SCHEDULER_QUALNAME
        )
        checks = {
            "expected_scheduler": result["resolved_classes"]["scheduler"]
            == expected_scheduler,
            "exactly_1000_requests": result["workload_summary"]["request_count"]
            == 1000,
            "all_400_cohorts_passed": len(result["cases"]) == 400
            and all(case["pass"] for case in result["cases"]),
        }
        if mode == "baseline":
            checks["stock_only"] = result["aggregate"]["device_model_steps"] == 0
        else:
            aggregate_record = result["aggregate"]
            checks.update(
                {
                    "device_route_exercised": aggregate_record["device_model_steps"]
                    > 0,
                    "no_host_device_owned_violation": not aggregate_record[
                        "host_device_owned_violations"
                    ],
                    "all_unsupported_requests_stayed_on_host": not aggregate_record[
                        "ineligible_device_requests"
                    ],
                    "all_device_results_valid": aggregate_record[
                        "invalid_device_results"
                    ]
                    == 0,
                    "all_kv_import_checksums_match": aggregate_record[
                        "kv_imports"
                    ]
                    > 0
                    and aggregate_record["kv_checksum_mismatches"] == 0,
                    "device_eos_short_epoch_exercised": aggregate_record[
                        "short_eos_device_epochs"
                    ]
                    > 0,
                    "device_owned_cancellation_exercised": aggregate_record[
                        "device_owned_cancellations"
                    ]
                    > 0,
                    "row_leases_stable_and_unique": aggregate_record[
                        "stable_request_leases"
                    ]
                    and aggregate_record["lease_aliases"] == 0,
                    "row_reuse_is_generation_checked": aggregate_record[
                        "row_reuses"
                    ]
                    > 0
                    and aggregate_record["nonmonotonic_row_reuse"] == 0,
                }
            )
        result["checks"] = checks
        result["pass"] = all(checks.values())
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
    names = set(baseline_cases) | set(cruise_cases)
    mismatches: dict[str, list[str]] = {}
    for name in sorted(names):
        baseline_case = baseline_cases.get(name, {})
        cruise_case = cruise_cases.get(name, {})
        fields = (
            "tokens",
            "terminal_finish_reasons",
            "terminal_stop_reasons",
            "cancelled_at",
            "final_request_state",
        )
        different = [
            field
            for field in fields
            if baseline_case.get(field) != cruise_case.get(field)
        ]
        if different:
            mismatches[name] = different
    checks = {
        "baseline_passed": baseline.get("pass") is True,
        "cruise_passed": cruise.get("pass") is True,
        "same_workload": baseline.get("workload_sha256")
        == cruise.get("workload_sha256"),
        "same_workload_summary": baseline.get("workload_summary")
        == cruise.get("workload_summary"),
        "same_case_set": set(baseline_cases) == set(cruise_cases),
        "all_1000_requests_equivalent": len(names) == 400 and not mismatches,
    }
    return {
        "schema_version": 1,
        "gate": "M1 1,000-request differential comparison",
        "baseline": str(baseline_path),
        "cruise": str(cruise_path),
        "mismatches": mismatches,
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
        default=Path(__file__).with_name("workload.json"),
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
            workload=load_workload(args.cases.resolve(strict=True)),
        )
    write_result(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
