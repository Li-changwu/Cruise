#!/usr/bin/env python3
"""Profile the Cruise Device graph without a colocated vLLM EngineCore."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
    EpochCommitState,
    ResidentEpochPlan,
    ResidentEpochRequest,
)
from vllm_ascend_resident_epoch.sidecar_backend import (
    GRAPH_BATCH_SIZE,
    SidecarDataFlowEngine,
)


EVIDENCE_LIMITATION = (
    "Post-warmup Device graph timeline attribution only; this sidecar-only "
    "run is not service-level performance evidence."
)


def _require_inside_runtime(path: Path, runtime_dir: Path) -> Path:
    resolved = path.resolve()
    if resolved != runtime_dir and runtime_dir not in resolved.parents:
        raise ValueError(f"profiler barrier must be inside runtime-dir: {resolved}")
    return resolved


def _build_plan(epoch_index: int, steps: int) -> ResidentEpochPlan:
    fixed_epoch_graph = os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_K6") == "1"
    blocks_per_request = 1 if fixed_epoch_graph else 2
    requests = tuple(
        ResidentEpochRequest(
            req_id=f"epoch-{epoch_index}-row-{row}",
            row=row,
            generation=epoch_index * GRAPH_BATCH_SIZE + row + 1,
            token_id=9707,
            position=0,
            sequence_length=1,
            eos_token_id=151645,
            scheduler_block_ids=tuple(
                row * blocks_per_request + local_block
                for local_block in range(blocks_per_request)
            ),
            device_block_ids=tuple(
                row * blocks_per_request + local_block
                for local_block in range(blocks_per_request)
            ),
            state_owner="device",
            kv_import_required=False,
        )
        for row in range(GRAPH_BATCH_SIZE)
    )
    plan = ResidentEpochPlan(
        version=CONTRACT_VERSION,
        graph_batch_size=GRAPH_BATCH_SIZE,
        max_steps=steps,
        logical_capacity=8,
        requests=requests,
        active_mask=(1,) * GRAPH_BATCH_SIZE,
        graph_variant=0x10 if fixed_epoch_graph else 0,
    )
    plan.validate()
    return plan


def _wait_for_file(
    path: Path,
    engine: SidecarDataFlowEngine,
    *,
    timeout_seconds: float,
    description: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        return_code = engine.process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"sidecar exited with status {return_code} while waiting for "
                f"{description}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(0.1)


def _warmup_record(output: Any) -> dict[str, Any]:
    return {
        "status": int(output.status),
        "model_calls": int(output.model_calls),
        "feed_calls": int(output.feed_calls),
        "fetch_calls": int(output.fetch_calls),
        "commit_state": int(output.commit_state),
        "wall_us": int(output.wall_us),
        "native_cpu_us": int(output.native_cpu_us),
    }


def _epoch_record(
    epoch_index: int, plan: ResidentEpochPlan, output: Any
) -> dict[str, Any]:
    token_ids = {
        request.req_id: [int(token) for token in output.token_ids[request.req_id]]
        for request in plan.requests
    }
    return {
        "epoch_index": epoch_index,
        "status": int(output.status),
        "model_calls": int(output.model_calls),
        "feed_calls": int(output.feed_calls),
        "fetch_calls": int(output.fetch_calls),
        "commit_state": int(output.commit_state),
        "wall_us": int(output.wall_us),
        "native_cpu_us": int(output.native_cpu_us),
        "row_generations": [int(value) for value in output.row_generations],
        "tokens_per_row": {
            request.req_id: len(token_ids[request.req_id])
            for request in plan.requests
        },
        "token_ids": token_ids,
        "kv_imported": bool(output.kv_imported),
    }


def _validate_epoch(
    plan: ResidentEpochPlan, output: Any, *, epoch_index: int
) -> None:
    if output.status != 0:
        raise RuntimeError(f"epoch {epoch_index} returned status {output.status}")
    if output.commit_state != EpochCommitState.COMMITTED:
        raise RuntimeError(f"epoch {epoch_index} was not committed")
    expected_model_calls = 1 if plan.graph_variant == 0x10 else plan.max_steps
    if output.model_calls != expected_model_calls:
        raise RuntimeError(
            f"epoch {epoch_index} used {output.model_calls} model calls, "
            f"expected {expected_model_calls}"
        )
    if output.feed_calls != 1 or output.fetch_calls != 1:
        raise RuntimeError(
            f"epoch {epoch_index} did not use exactly one Feed and one Fetch"
        )
    if output.row_generations != plan.row_generations:
        raise RuntimeError(f"epoch {epoch_index} generation acknowledgement mismatch")
    if output.kv_imported:
        raise RuntimeError(f"epoch {epoch_index} unexpectedly imported Device KV")
    for request in plan.requests:
        count = len(output.token_ids.get(request.req_id, ()))
        if count != plan.max_steps:
            raise RuntimeError(
                f"epoch {epoch_index} row {request.row} returned {count} tokens, "
                f"expected {plan.max_steps}"
            )


def run_profile(
    *,
    runtime_dir: Path,
    ready_file: Path,
    start_file: Path,
    workload_done_file: Path,
    release_file: Path,
    output_file: Path,
    epochs: int,
    steps: int,
    barrier_timeout_seconds: float = 600.0,
    engine_factory: Callable[[], SidecarDataFlowEngine] = SidecarDataFlowEngine,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not 1 <= steps <= 8:
        raise ValueError("steps must be between one and eight")

    runtime_dir = runtime_dir.resolve()
    barriers = tuple(
        _require_inside_runtime(path, runtime_dir)
        for path in (ready_file, start_file, workload_done_file, release_file)
    )
    ready_file, start_file, workload_done_file, release_file = barriers
    if any(path.exists() for path in barriers):
        raise FileExistsError("profiler barrier file already exists")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for child in ("cache", "cann-logs", "tmp"):
        (runtime_dir / child).mkdir(exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": "M4a Cruise sidecar-only Device graph attribution",
        "mode": "cruise-sidecar-only",
        "execution_scope": {
            "engine_core_colocated": False,
            "device_kv_import": False,
            "first_epoch_device_kv_import": False,
            "evidence_limitation": EVIDENCE_LIMITATION,
        },
        "runtime_dir": str(runtime_dir),
        "harness_pid": os.getpid(),
        "epochs_requested": epochs,
        "steps_per_epoch": steps,
        "fixed_epoch_graph": os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_K6") == "1",
        "warmup": None,
        "profiler_lifecycle": "post-warmup-in-process",
        "profiler_started": False,
        "profiler_stopped": False,
        "epochs": [],
        "checks": {},
        "pass": False,
    }
    engine: SidecarDataFlowEngine | None = None
    profiler_active = False
    started_ns = time.perf_counter_ns()
    cpu_started_ns = time.process_time_ns()
    try:
        engine = engine_factory()
        result["sidecar_pid"] = int(engine.process.pid)
        warmup = engine.warm_up()
        result["warmup"] = _warmup_record(warmup)
        if (
            warmup.status != 0
            or warmup.model_calls != 1
            or warmup.feed_calls != 1
            or warmup.fetch_calls != 1
            or warmup.commit_state != EpochCommitState.COMMITTED
        ):
            raise RuntimeError("sidecar warmup did not complete one Device step")

        ready_file.write_text(
            json.dumps(
                {
                    "runner_pid": os.getpid(),
                    "api_server_pid": os.getpid(),
                    "harness_pid": os.getpid(),
                    "sidecar_pid": engine.process.pid,
                    "mode": "cruise-sidecar-only",
                    "profiler_lifecycle": "post-warmup-in-process",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _wait_for_file(
            start_file,
            engine,
            timeout_seconds=barrier_timeout_seconds,
            description="profiler start barrier",
        )
        engine.start_profiling()
        profiler_active = True
        result["profiler_started"] = True

        workload_started_ns = time.perf_counter_ns()
        workload_cpu_started_ns = time.process_time_ns()
        for epoch_index in range(epochs):
            plan = _build_plan(epoch_index, steps)
            output = engine.execute(plan)
            _validate_epoch(plan, output, epoch_index=epoch_index)
            result["epochs"].append(_epoch_record(epoch_index, plan, output))
        result["workload_wall_ms"] = (
            time.perf_counter_ns() - workload_started_ns
        ) / 1_000_000
        result["workload_cpu_ms"] = (
            time.process_time_ns() - workload_cpu_started_ns
        ) / 1_000_000
        engine.stop_profiling()
        profiler_active = False
        result["profiler_stopped"] = True

        checks = {
            "warmup_complete": result["warmup"]["status"] == 0,
            "all_epochs_complete": len(result["epochs"]) == epochs,
            "all_status_zero": all(item["status"] == 0 for item in result["epochs"]),
            "all_model_calls_match_graph": all(
                item["model_calls"]
                == (1 if result["fixed_epoch_graph"] else steps)
                for item in result["epochs"]
            ),
            "one_feed_per_epoch": all(
                item["feed_calls"] == 1 for item in result["epochs"]
            ),
            "one_fetch_per_epoch": all(
                item["fetch_calls"] == 1 for item in result["epochs"]
            ),
            "six_tokens_per_row": steps == 6
            and all(
                count == 6
                for item in result["epochs"]
                for count in item["tokens_per_row"].values()
            ),
            "no_device_kv_import": all(
                not item["kv_imported"] for item in result["epochs"]
            ),
            "no_engine_core_colocated": True,
            "post_warmup_profiler_started": result["profiler_started"],
            "post_warmup_profiler_stopped": result["profiler_stopped"],
        }
        result["checks"] = checks
        result["pass"] = all(checks.values())
        workload_done_file.write_text(
            json.dumps(
                {
                    "runner_pid": os.getpid(),
                    "api_server_pid": os.getpid(),
                    "harness_pid": os.getpid(),
                    "sidecar_pid": engine.process.pid,
                    "epochs": epochs,
                    "steps_per_epoch": steps,
                    "profiler_started": result["profiler_started"],
                    "profiler_stopped": result["profiler_stopped"],
                    "pass": result["pass"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _wait_for_file(
            release_file,
            engine,
            timeout_seconds=barrier_timeout_seconds,
            description="profiler release barrier",
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["pass"] = False
    finally:
        if profiler_active and engine is not None:
            try:
                engine.stop_profiling()
                result["profiler_stopped"] = True
            except Exception as exc:
                result["profiler_stop_error"] = f"{type(exc).__name__}: {exc}"
                result["pass"] = False
        if engine is not None:
            try:
                engine.close()
                result["sidecar_returncode"] = int(engine.process.returncode)
                clean_exit = engine.process.returncode == 0
                result["checks"]["clean_sidecar_exit"] = clean_exit
                result["pass"] = result["pass"] and clean_exit
            except Exception as exc:
                result["shutdown_error"] = f"{type(exc).__name__}: {exc}"
                result["pass"] = False
        result["total_wall_ms"] = (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000
        result["total_cpu_ms"] = (
            time.process_time_ns() - cpu_started_ns
        ) / 1_000_000
        output_file.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--profile-ready-file", required=True, type=Path)
    parser.add_argument("--profile-start-file", required=True, type=Path)
    parser.add_argument("--profile-workload-done-file", required=True, type=Path)
    parser.add_argument("--profile-release-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    result = run_profile(
        runtime_dir=args.runtime_dir,
        ready_file=args.profile_ready_file,
        start_file=args.profile_start_file,
        workload_done_file=args.profile_workload_done_file,
        release_file=args.profile_release_file,
        output_file=args.output,
        epochs=args.epochs,
        steps=args.steps,
    )
    return int(not result["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
