from dataclasses import dataclass
from typing import Any, Literal


PLAN_ATTR = "_ascend_resident_epoch_plan"
RESULT_ATTR = "_ascend_resident_epoch_result"
CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ResidentEpochRequest:
    req_id: str
    row: int
    token_id: int
    position: int
    sequence_length: int
    eos_token_id: int
    scheduler_block_ids: tuple[int, ...]
    device_block_ids: tuple[int, int]


@dataclass(frozen=True)
class ResidentEpochPlan:
    version: int
    graph_batch_size: Literal[1, 2, 4]
    max_steps: int
    logical_capacity: int
    requests: tuple[ResidentEpochRequest, ...]
    active_mask: tuple[int, ...]
    sampling_mode: int = 0
    graph_variant: int = 0

    @property
    def req_ids(self) -> tuple[str, ...]:
        return tuple(request.req_id for request in self.requests)

    def validate(self) -> None:
        if self.version != CONTRACT_VERSION:
            raise ValueError(f"unsupported resident epoch contract {self.version}")
        if self.graph_batch_size not in (1, 2, 4):
            raise ValueError("graph batch size must be 1, 2, or 4")
        if not 1 <= len(self.requests) <= self.graph_batch_size:
            raise ValueError("request count does not fit the static graph")
        if len(self.active_mask) != self.graph_batch_size:
            raise ValueError("active mask does not match graph batch size")
        if tuple(self.active_mask[: len(self.requests)]) != (1,) * len(
            self.requests
        ):
            raise ValueError("scheduled request rows must be active")
        if any(self.active_mask[len(self.requests) :]):
            raise ValueError("padding rows must be inactive")
        if len(set(self.req_ids)) != len(self.requests):
            raise ValueError("request IDs must be unique")
        for row, request in enumerate(self.requests):
            if request.row != row:
                raise ValueError("request rows must be dense and ordered")
            if request.sequence_length != request.position + 1:
                raise ValueError("sequence length and position disagree")
            if request.position + self.max_steps > self.logical_capacity:
                raise ValueError("epoch exceeds logical capacity")


@dataclass(frozen=True)
class ResidentEpochResult:
    version: int
    route: Literal["device", "host_fallback"]
    status: int
    model_calls: int
    computed_steps: dict[str, int]
    fallback_safe: bool = False
    feed_calls: int = 1
    fetch_calls: int = 1
    wall_us: int = 0

    def validate_against(
        self,
        plan: ResidentEpochPlan,
        sampled_token_ids: dict[str, list[int]],
    ) -> None:
        if self.version != plan.version:
            raise ValueError("resident epoch result version mismatch")
        if set(self.computed_steps) != set(plan.req_ids):
            raise ValueError("resident epoch result request set mismatch")
        if set(sampled_token_ids) != set(plan.req_ids):
            raise ValueError("resident epoch output request set mismatch")
        if self.route == "device" and self.status != 0:
            raise ValueError("failed device execution cannot be committed")
        if self.route == "device" and (
            self.feed_calls != 1 or self.fetch_calls != 1
        ):
            raise ValueError("device epoch must use exactly one Feed and one Fetch")
        if self.route == "host_fallback" and not self.fallback_safe:
            raise ValueError("host fallback must be declared input-preserving")
        if self.route == "host_fallback" and (
            self.feed_calls != 0 or self.fetch_calls != 0
        ):
            raise ValueError("Host fallback result cannot claim DataFlow calls")
        for request in plan.requests:
            steps = self.computed_steps[request.req_id]
            tokens = sampled_token_ids[request.req_id]
            if steps < 1 or steps > plan.max_steps:
                raise ValueError("computed step count is outside the epoch")
            if len(tokens) != steps:
                raise ValueError("sampled token count and computed steps disagree")
            if self.route == "host_fallback" and steps != 1:
                raise ValueError("host fallback must retain one-step vLLM semantics")
            if self.route == "device" and steps < plan.max_steps:
                if not tokens or tokens[-1] != request.eos_token_id:
                    raise ValueError("a short device epoch must terminate at EOS")
        if self.route == "device" and self.model_calls != max(
            self.computed_steps.values()
        ):
            raise ValueError("device model-call count disagrees with executed steps")


def attach_plan(scheduler_output: Any, plan: ResidentEpochPlan) -> None:
    plan.validate()
    setattr(scheduler_output, PLAN_ATTR, plan)


def get_plan(scheduler_output: Any) -> ResidentEpochPlan | None:
    value = getattr(scheduler_output, PLAN_ATTR, None)
    if value is not None and not isinstance(value, ResidentEpochPlan):
        raise TypeError("invalid resident epoch plan attached to SchedulerOutput")
    return value


def attach_result(model_runner_output: Any, result: ResidentEpochResult) -> None:
    setattr(model_runner_output, RESULT_ATTR, result)


def get_result(model_runner_output: Any) -> ResidentEpochResult | None:
    value = getattr(model_runner_output, RESULT_ATTR, None)
    if value is not None and not isinstance(value, ResidentEpochResult):
        raise TypeError("invalid resident epoch result attached to ModelRunnerOutput")
    return value
