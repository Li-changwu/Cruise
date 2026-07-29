from dataclasses import dataclass

from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
    EpochCommitState,
    ResidentEpochPlan,
    ResidentEpochRequest,
    ResidentEpochResult,
    attach_plan,
    attach_result,
)
from vllm_ascend_resident_epoch.scheduler import ResidentEpochScheduler


@dataclass
class FakeRequest:
    num_computed_tokens: int
    num_tokens: int


def make_scheduler():
    scheduler = object.__new__(ResidentEpochScheduler)
    scheduler.requests = {"r0": FakeRequest(num_computed_tokens=1, num_tokens=1)}
    return scheduler


def make_plan(max_steps=4):
    return ResidentEpochPlan(
        version=CONTRACT_VERSION,
        graph_batch_size=1,
        max_steps=max_steps,
        logical_capacity=8,
        requests=(
            ResidentEpochRequest(
                req_id="r0",
                row=0,
                generation=1,
                token_id=1,
                position=0,
                sequence_length=1,
                eos_token_id=9,
                scheduler_block_ids=(4,),
                device_block_ids=(0, 1),
            ),
        ),
        active_mask=(1,),
    )


def test_device_epoch_advances_computed_tokens_by_actual_steps():
    scheduler = make_scheduler()
    scheduler_output = type("SchedulerOutput", (), {})()
    scheduler_output.num_scheduled_tokens = {"r0": 1}
    plan = make_plan()
    attach_plan(scheduler_output, plan)

    output = ModelRunnerOutput(
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        sampled_token_ids=[[2, 3, 4, 5]],
    )
    attach_result(
        output,
        ResidentEpochResult(
            version=CONTRACT_VERSION,
            route="device",
            status=0,
            model_calls=4,
            computed_steps={"r0": 4},
            row_generations=plan.row_generations,
        ),
    )

    scheduler._apply_resident_epoch_accounting(
        scheduler_output,
        output,
        plan,
        getattr(output, "_ascend_resident_epoch_result"),
    )
    assert scheduler.requests["r0"].num_computed_tokens == 4


def test_host_fallback_retains_normal_one_step_accounting():
    scheduler = make_scheduler()
    scheduler_output = type("SchedulerOutput", (), {})()
    scheduler_output.num_scheduled_tokens = {"r0": 1}
    plan = make_plan()
    attach_plan(scheduler_output, plan)
    output = ModelRunnerOutput(
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        sampled_token_ids=[[2]],
    )
    attach_result(
        output,
        ResidentEpochResult(
            version=CONTRACT_VERSION,
            route="host_fallback",
            status=0,
            model_calls=1,
            computed_steps={"r0": 1},
            commit_state=EpochCommitState.PREPARED,
            fallback_safe=True,
            feed_calls=0,
            fetch_calls=0,
        ),
    )
    scheduler._apply_resident_epoch_accounting(
        scheduler_output,
        output,
        plan,
        getattr(output, "_ascend_resident_epoch_result"),
    )
    assert scheduler.requests["r0"].num_computed_tokens == 1
