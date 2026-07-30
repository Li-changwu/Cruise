from __future__ import annotations

from typing import Any

from .backend import load_backend_from_env
from .contract import (
    CONTRACT_VERSION,
    EpochCommitState,
    ResidentEpochExecutionError,
    ResidentEpochResult,
    attach_result,
    get_plan,
)
from .kv_transfer import capture_kv_snapshot
from .triton_compat import ensure_triton_ascend_runtime


def _execute_model_with_fallback(
    worker: Any,
    scheduler_output: Any,
    original_execute_model: Any,
):
    plan = get_plan(scheduler_output)
    if plan is None:
        return original_execute_model(worker, scheduler_output)

    backend = getattr(worker, "_resident_epoch_backend", None)
    if backend is None:
        try:
            backend = load_backend_from_env()
        except Exception:
            if not plan.host_replay_safe:
                raise
            output = original_execute_model(worker, scheduler_output)
            return attach_host_fallback_result(output, plan.req_ids)
        worker._resident_epoch_backend = backend
    try:
        snapshot = (
            capture_kv_snapshot(worker, plan)
            if any(request.kv_import_required for request in plan.requests)
            else None
        )
        if snapshot is None:
            return backend.execute(plan)
        return backend.execute(plan, snapshot=snapshot)
    except ResidentEpochExecutionError as exc:
        if not exc.input_preserving or not plan.host_replay_safe:
            raise
        output = original_execute_model(worker, scheduler_output)
        return attach_host_fallback_result(output, plan.req_ids)


def register() -> None:
    ensure_triton_ascend_runtime()
    from vllm_ascend.worker.worker import NPUWorker

    if hasattr(NPUWorker, "_resident_epoch_original_execute_model"):
        return

    original_execute_model = NPUWorker.execute_model
    original_shutdown = NPUWorker.shutdown

    def execute_model(self: Any, scheduler_output: Any):
        return _execute_model_with_fallback(
            self, scheduler_output, original_execute_model
        )

    def shutdown(self: Any) -> None:
        backend = getattr(self, "_resident_epoch_backend", None)
        try:
            if backend is not None:
                backend.close()
        finally:
            self._resident_epoch_backend = None
            original_shutdown(self)

    NPUWorker.execute_model = execute_model
    NPUWorker.shutdown = shutdown
    NPUWorker._resident_epoch_original_execute_model = original_execute_model
    NPUWorker._resident_epoch_original_shutdown = original_shutdown


def attach_host_fallback_result(
    output: Any, req_ids: tuple[str, ...]
) -> ModelRunnerOutput:
    """Attach accounting metadata after a proven zero-model-call fallback."""
    from vllm.v1.outputs import ModelRunnerOutput

    if not isinstance(output, ModelRunnerOutput):
        raise TypeError("resident epoch fallback requires synchronous ModelRunnerOutput")
    attach_result(
        output,
        ResidentEpochResult(
            version=CONTRACT_VERSION,
            route="host_fallback",
            status=0,
            model_calls=1,
            computed_steps={req_id: 1 for req_id in req_ids},
            commit_state=EpochCommitState.PREPARED,
            fallback_safe=True,
            feed_calls=0,
            fetch_calls=0,
        ),
    )
    return output
