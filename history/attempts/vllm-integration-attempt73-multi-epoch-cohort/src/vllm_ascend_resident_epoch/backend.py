from dataclasses import dataclass
from importlib import import_module
import os
from typing import Protocol

from vllm.v1.outputs import ModelRunnerOutput

from .contract import (
    CONTRACT_VERSION,
    ResidentEpochPlan,
    ResidentEpochResult,
    attach_result,
)


@dataclass(frozen=True)
class NativeEpochOutput:
    status: int
    model_calls: int
    token_ids: dict[str, list[int]]
    row_generations: tuple[int, ...]
    feed_calls: int = 1
    fetch_calls: int = 1
    wall_us: int = 0


@dataclass(frozen=True)
class NativeWarmupOutput:
    status: int
    model_calls: int
    feed_calls: int
    fetch_calls: int
    wall_us: int


class NativeEpochEngine(Protocol):
    def warm_up(self) -> NativeWarmupOutput: ...

    def execute(self, plan: ResidentEpochPlan) -> NativeEpochOutput: ...


class ResidentEpochBackend:
    def __init__(self, engine: NativeEpochEngine):
        self.engine = engine

    def warm_up(self) -> NativeWarmupOutput:
        output = self.engine.warm_up()
        if output.status != 0:
            raise RuntimeError(f"native resident warmup returned {output.status}")
        if (
            output.model_calls != 1
            or output.feed_calls != 1
            or output.fetch_calls != 1
        ):
            raise RuntimeError("resident warmup must execute one complete decoder step")
        return output

    def execute(self, plan: ResidentEpochPlan) -> ModelRunnerOutput:
        plan.validate()
        native_output = self.engine.execute(plan)
        if native_output.status != 0:
            raise RuntimeError(
                "native resident epoch returned status "
                f"{native_output.status}; worker must perform safe fallback"
            )
        if set(native_output.token_ids) != set(plan.req_ids):
            raise RuntimeError("native resident epoch returned the wrong request set")

        req_ids = list(plan.req_ids)
        sampled = [native_output.token_ids[req_id] for req_id in req_ids]
        result = ResidentEpochResult(
            version=CONTRACT_VERSION,
            route="device",
            status=0,
            model_calls=native_output.model_calls,
            computed_steps={
                req_id: len(native_output.token_ids[req_id]) for req_id in req_ids
            },
            row_generations=native_output.row_generations,
            feed_calls=native_output.feed_calls,
            fetch_calls=native_output.fetch_calls,
            wall_us=native_output.wall_us,
        )
        result.validate_against(
            plan,
            {req_id: native_output.token_ids[req_id] for req_id in req_ids},
        )
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
        )
        attach_result(output, result)
        return output


def load_backend_from_env() -> ResidentEpochBackend:
    target = os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY")
    if not target or ":" not in target:
        raise RuntimeError(
            "VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY must name "
            "a module:factory for the native DataFlow backend"
        )
    module_name, factory_name = target.split(":", 1)
    factory = getattr(import_module(module_name), factory_name)
    engine = factory()
    return ResidentEpochBackend(engine)
