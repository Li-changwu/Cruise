from __future__ import annotations

from typing import Any

from .backend import load_backend_from_env
from .contract import (
    CONTRACT_VERSION,
    ResidentEpochResult,
    attach_result,
    get_plan,
)


def register() -> None:
    from vllm_ascend.worker.worker import NPUWorker

    if hasattr(NPUWorker, "_resident_epoch_original_execute_model"):
        return

    original_execute_model = NPUWorker.execute_model

    def execute_model(self: Any, scheduler_output: Any):
        plan = get_plan(scheduler_output)
        if plan is None:
            return original_execute_model(self, scheduler_output)

        backend = getattr(self, "_resident_epoch_backend", None)
        if backend is None:
            backend = load_backend_from_env()
            self._resident_epoch_backend = backend
        try:
            return backend.execute(plan)
        except Exception:
            # The G4 controller proves only pre-execution input-preserving
            # fallback. Until the native bridge reports that distinction, do
            # not guess whether Host replay is safe.
            raise

    NPUWorker.execute_model = execute_model
    NPUWorker._resident_epoch_original_execute_model = original_execute_model


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
            fallback_safe=True,
            feed_calls=0,
            fetch_calls=0,
        ),
    )
    return output
