from types import SimpleNamespace

import pytest
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
    EpochCommitState,
    ResidentEpochExecutionError,
    ResidentEpochPlan,
    ResidentEpochRequest,
    attach_plan,
    get_result,
)
from vllm_ascend_resident_epoch.plugin import _execute_model_with_fallback


def _plan(owner: str) -> ResidentEpochPlan:
    return ResidentEpochPlan(
        version=CONTRACT_VERSION,
        graph_batch_size=1,
        max_steps=1,
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
                state_owner=owner,
            ),
        ),
        active_mask=(1,),
    )


class _FailingBackend:
    def __init__(self, state: EpochCommitState) -> None:
        self.state = state

    def execute(self, plan: ResidentEpochPlan):
        raise ResidentEpochExecutionError(
            "injected failure", commit_state=self.state
        )


def _host_output(*_args) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        sampled_token_ids=[[7]],
    )


def _execute(owner: str, state: EpochCommitState):
    scheduler_output = SimpleNamespace()
    attach_plan(scheduler_output, _plan(owner))
    worker = SimpleNamespace(_resident_epoch_backend=_FailingBackend(state))
    return _execute_model_with_fallback(worker, scheduler_output, _host_output)


def test_prepared_failure_replays_when_host_still_owns_state():
    output = _execute("host", EpochCommitState.PREPARED)
    result = get_result(output)
    assert result is not None
    assert result.route == "host_fallback"
    assert result.commit_state == EpochCommitState.PREPARED
    assert result.fallback_safe


@pytest.mark.parametrize(
    "owner,state",
    [
        ("device", EpochCommitState.PREPARED),
        ("host", EpochCommitState.EXECUTING),
        ("host", EpochCommitState.COMMITTED),
    ],
)
def test_ambiguous_or_device_owned_failure_never_replays(owner, state):
    with pytest.raises(ResidentEpochExecutionError) as caught:
        _execute(owner, state)
    assert caught.value.commit_state == state
