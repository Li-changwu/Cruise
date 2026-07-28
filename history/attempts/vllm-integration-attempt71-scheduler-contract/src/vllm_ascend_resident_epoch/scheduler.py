from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput

from .config import ResidentEpochConfig
from .contract import (
    CONTRACT_VERSION,
    ResidentEpochPlan,
    ResidentEpochRequest,
    attach_plan,
    get_plan,
    get_result,
)
from .eligibility import request_rejection_reason


class ResidentEpochScheduler(Scheduler):
    """Scheduler extension for the fixed-shape G4 resident epoch.

    The native graph currently owns a complete request only when a one-token
    prompt can finish in one bounded epoch. Unsupported requests retain the
    ordinary vLLM scheduling and execution path.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._resident_epoch_config = ResidentEpochConfig.from_env()
        self._resident_epoch_last_rejection: str | None = None
        self._resident_epoch_last_plan: ResidentEpochPlan | None = None
        self._resident_epoch_last_result: Any | None = None
        if self.vllm_config.speculative_config is not None:
            raise ValueError("resident epoch scheduler cannot be combined with speculative decoding")
        if self.scheduler_config.async_scheduling:
            raise ValueError("resident epoch scheduler requires synchronous scheduling")
        if self.vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("resident epoch scheduler currently requires tensor parallel size 1")
        if self.vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("resident epoch scheduler currently requires pipeline parallel size 1")
        model_reason = self._native_model_rejection_reason()
        if model_reason is not None:
            raise ValueError(f"resident epoch native graph mismatch: {model_reason}")

    def _ensure_test_config(self) -> ResidentEpochConfig:
        config = getattr(self, "_resident_epoch_config", None)
        if config is None:
            config = ResidentEpochConfig()
            self._resident_epoch_config = config
            self._resident_epoch_last_rejection = None
        return config

    def _native_model_rejection_reason(self) -> str | None:
        model_config = self.vllm_config.model_config
        hf_config = model_config.hf_config
        expected = {
            "model_type": "qwen2",
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "vocab_size": 152064,
        }
        for field, expected_value in expected.items():
            if getattr(hf_config, field, None) != expected_value:
                return f"{field}={getattr(hf_config, field, None)!r}"
        return None

    def schedule(self):
        self._ensure_test_config()
        if self.num_lookahead_tokens != 0:
            raise RuntimeError("resident epoch lookahead conflicts with another feature")
        scheduler_output = super().schedule()

        plan, rejection = self._make_resident_epoch_plan(scheduler_output)
        self._resident_epoch_last_rejection = rejection
        self._resident_epoch_last_plan = plan
        self._resident_epoch_last_result = None
        if plan is not None:
            attach_plan(scheduler_output, plan)
        return scheduler_output

    def _make_resident_epoch_plan(
        self, scheduler_output: Any
    ) -> tuple[ResidentEpochPlan | None, str | None]:
        config = self._ensure_test_config()
        req_ids = tuple(scheduler_output.num_scheduled_tokens)
        if not req_ids:
            return None, "empty-schedule"
        if len(req_ids) > config.max_batch_size:
            return None, "batch-too-large"
        if any(
            scheduler_output.num_scheduled_tokens[req_id] != 1
            for req_id in req_ids
        ):
            return None, "not-single-token-decode"
        if scheduler_output.scheduled_spec_decode_tokens:
            return None, "speculative-decode"
        if scheduler_output.scheduled_encoder_inputs:
            return None, "encoder-input"
        if scheduler_output.preempted_req_ids:
            return None, "preemption"
        if self.connector is not None or self.ec_connector is not None:
            return None, "cache-connector"
        if self.block_size != config.block_size:
            return None, "block-size"
        if set(req_ids) != {request.request_id for request in self.running}:
            return None, "partial-running-set"

        requests = [self.requests[req_id] for req_id in req_ids]
        remaining_steps: list[int] = []
        for request in requests:
            reason = request_rejection_reason(request)
            if reason is not None:
                return None, reason
            if request.num_prompt_tokens != 1 or request.num_output_tokens != 0:
                return None, "native-kv-not-owned"
            if request.num_computed_tokens != request.num_tokens:
                return None, "computed-token-accounting"
            remaining = request.max_tokens - request.num_output_tokens
            if remaining < 1 or remaining > config.max_steps:
                return None, "epoch-does-not-finish-request"
            if remaining not in (1, 2, 4, 8):
                return None, "unsupported-epoch-steps"
            remaining_steps.append(remaining)
        if len(set(remaining_steps)) != 1:
            return None, "heterogeneous-max-steps"

        epoch_steps = remaining_steps[0]
        # One B=4 GraphPp instance serves all admitted batch sizes. Loading
        # B=1/B=2/B=4 together would duplicate the 15 GB external-weight model.
        graph_batch_size = 4
        request_plans: list[ResidentEpochRequest] = []
        for row, request in enumerate(requests):
            position = request.num_computed_tokens - 1
            if position < 0 or position + epoch_steps > config.logical_capacity:
                return None, "logical-capacity"
            block_ids = self.kv_cache_manager.get_blocks(
                request.request_id
            ).get_block_ids()
            if len(block_ids) != 1:
                return None, "multiple-kv-cache-groups"
            flat_block_ids = tuple(block_ids[0])
            if not flat_block_ids:
                return None, "missing-kv-block"
            params = request.sampling_params
            assert params is not None and params.eos_token_id is not None
            request_plans.append(
                ResidentEpochRequest(
                    req_id=request.request_id,
                    row=row,
                    token_id=int(request.all_token_ids[position]),
                    position=position,
                    sequence_length=position + 1,
                    eos_token_id=int(params.eos_token_id),
                    scheduler_block_ids=flat_block_ids,
                    device_block_ids=(
                        row * config.blocks_per_request,
                        row * config.blocks_per_request + 1,
                    ),
                )
            )

        plan = ResidentEpochPlan(
            version=CONTRACT_VERSION,
            graph_batch_size=graph_batch_size,
            max_steps=epoch_steps,
            logical_capacity=config.logical_capacity,
            requests=tuple(request_plans),
            active_mask=(1,) * len(request_plans)
            + (0,) * (graph_batch_size - len(request_plans)),
        )
        plan.validate()
        return plan, None

    def update_from_output(
        self,
        scheduler_output: Any,
        model_runner_output: ModelRunnerOutput,
    ):
        plan = get_plan(scheduler_output)
        if plan is None:
            if get_result(model_runner_output) is not None:
                raise RuntimeError("resident epoch result has no scheduler plan")
            return super().update_from_output(scheduler_output, model_runner_output)

        result = get_result(model_runner_output)
        if result is None:
            raise RuntimeError("resident epoch plan did not receive execution metadata")
        self._resident_epoch_last_result = result
        self._apply_resident_epoch_accounting(
            scheduler_output, model_runner_output, plan, result
        )
        return super().update_from_output(scheduler_output, model_runner_output)

    def _apply_resident_epoch_accounting(
        self,
        scheduler_output: Any,
        model_runner_output: ModelRunnerOutput,
        plan: ResidentEpochPlan,
        result: Any,
    ) -> None:
        sampled_by_req = {
            req_id: list(
                model_runner_output.sampled_token_ids[
                    model_runner_output.req_id_to_index[req_id]
                ]
            )
            for req_id in plan.req_ids
        }
        result.validate_against(plan, sampled_by_req)

        extras: dict[str, int] = {}
        for req_id, steps in result.computed_steps.items():
            request = self.requests[req_id]
            extra = steps - scheduler_output.num_scheduled_tokens[req_id]
            if extra < 0:
                raise RuntimeError("resident epoch computed fewer than scheduled tokens")
            extras[req_id] = extra
            if request.num_computed_tokens + extra > request.num_tokens + len(
                sampled_by_req[req_id]
            ):
                raise RuntimeError("resident epoch would over-advance computed tokens")

        for req_id, extra in extras.items():
            self.requests[req_id].num_computed_tokens += extra
