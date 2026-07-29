from types import SimpleNamespace

import pytest

from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
    EpochCommitState,
    ResidentEpochExecutionError,
    ResidentEpochPlan,
    ResidentEpochRequest,
    ResidentEpochResult,
)
from vllm_ascend_resident_epoch.eligibility import request_rejection_reason


def make_request(**overrides):
    params = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        min_tokens=0,
        ignore_eos=False,
        eos_token_id=151645,
        stop=[],
        stop_token_ids=[],
        logprobs=None,
        prompt_logprobs=None,
        logit_bias=None,
        allowed_token_ids=None,
        bad_words=[],
        repetition_detection=None,
        extra_args=None,
    )
    request = SimpleNamespace(
        pooling_params=None,
        mm_features=[],
        lora_request=None,
        prompt_embeds=None,
        use_structured_output=False,
        sampling_params=params,
    )
    for key, value in overrides.items():
        if hasattr(params, key):
            setattr(params, key, value)
        else:
            setattr(request, key, value)
    return request


def make_plan() -> ResidentEpochPlan:
    return ResidentEpochPlan(
        version=CONTRACT_VERSION,
        graph_batch_size=1,
        max_steps=4,
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
                scheduler_block_ids=(17,),
                device_block_ids=(0, 1),
            ),
        ),
        active_mask=(1,),
    )


def test_greedy_request_is_eligible():
    assert request_rejection_reason(make_request()) is None


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("temperature", 0.8, "non-greedy-temperature"),
        ("top_k", 8, "unsupported-sampling-filter"),
        ("presence_penalty", 0.1, "sampling-penalty"),
        ("min_tokens", 2, "min-tokens"),
        ("ignore_eos", True, "ignore-eos"),
        ("stop_token_ids", [7], "extra-stop-condition"),
        ("logprobs", 1, "logprobs"),
        ("allowed_token_ids", [1, 2], "logit-processor"),
    ],
)
def test_unsupported_semantics_are_rejected(field, value, reason):
    assert request_rejection_reason(make_request(**{field: value})) == reason


def test_short_device_result_requires_eos():
    plan = make_plan()
    result = ResidentEpochResult(
        version=CONTRACT_VERSION,
        route="device",
        status=0,
        model_calls=2,
        computed_steps={"r0": 2},
        row_generations=plan.row_generations,
    )
    with pytest.raises(ValueError, match="terminate at EOS"):
        result.validate_against(plan, {"r0": [3, 4]})
    result.validate_against(plan, {"r0": [3, 9]})


def test_host_fallback_keeps_one_step_accounting():
    plan = make_plan()
    result = ResidentEpochResult(
        version=CONTRACT_VERSION,
        route="host_fallback",
        status=0,
        model_calls=1,
        computed_steps={"r0": 1},
        commit_state=EpochCommitState.PREPARED,
        fallback_safe=True,
        feed_calls=0,
        fetch_calls=0,
    )
    result.validate_against(plan, {"r0": [3]})


@pytest.mark.parametrize(
    "state", [EpochCommitState.PREPARED, EpochCommitState.EXECUTING]
)
def test_device_result_requires_committed_state(state):
    plan = make_plan()
    result = ResidentEpochResult(
        version=CONTRACT_VERSION,
        route="device",
        status=0,
        model_calls=4,
        computed_steps={"r0": 4},
        row_generations=plan.row_generations,
        commit_state=state,
    )
    with pytest.raises(ValueError, match="not committed"):
        result.validate_against(plan, {"r0": [2, 3, 4, 5]})


def test_host_fallback_rejects_device_owned_request():
    base = make_plan()
    request = base.requests[0]
    plan = ResidentEpochPlan(
        version=base.version,
        graph_batch_size=base.graph_batch_size,
        max_steps=base.max_steps,
        logical_capacity=base.logical_capacity,
        requests=(
            ResidentEpochRequest(
                req_id=request.req_id,
                row=request.row,
                generation=request.generation,
                token_id=request.token_id,
                position=request.position,
                sequence_length=request.sequence_length,
                eos_token_id=request.eos_token_id,
                scheduler_block_ids=request.scheduler_block_ids,
                device_block_ids=request.device_block_ids,
                state_owner="device",
            ),
        ),
        active_mask=base.active_mask,
    )
    result = ResidentEpochResult(
        version=CONTRACT_VERSION,
        route="host_fallback",
        status=0,
        model_calls=1,
        computed_steps={"r0": 1},
        commit_state=EpochCommitState.PREPARED,
        fallback_safe=True,
        feed_calls=0,
        fetch_calls=0,
    )
    with pytest.raises(ValueError, match="Host no longer owns"):
        result.validate_against(plan, {"r0": [3]})


def test_only_prepared_execution_error_is_input_preserving():
    prepared = ResidentEpochExecutionError(
        "before Feed", commit_state=EpochCommitState.PREPARED
    )
    executing = ResidentEpochExecutionError(
        "Feed may have run", commit_state=EpochCommitState.EXECUTING
    )
    committed = ResidentEpochExecutionError(
        "output invalid", commit_state=EpochCommitState.COMMITTED
    )
    assert prepared.input_preserving
    assert not executing.input_preserving
    assert not committed.input_preserving


def test_kv_import_requires_host_ownership_and_commit_acknowledgement():
    base = make_plan()
    request = base.requests[0]
    importing_request = ResidentEpochRequest(
        req_id=request.req_id,
        row=request.row,
        generation=request.generation,
        token_id=request.token_id,
        position=request.position,
        sequence_length=request.sequence_length,
        eos_token_id=request.eos_token_id,
        scheduler_block_ids=request.scheduler_block_ids,
        device_block_ids=request.device_block_ids,
        state_owner="host",
        kv_import_required=True,
    )
    plan = ResidentEpochPlan(
        version=base.version,
        graph_batch_size=base.graph_batch_size,
        max_steps=base.max_steps,
        logical_capacity=base.logical_capacity,
        requests=(importing_request,),
        active_mask=base.active_mask,
    )
    plan.validate()
    result = ResidentEpochResult(
        version=CONTRACT_VERSION,
        route="device",
        status=0,
        model_calls=4,
        computed_steps={"r0": 4},
        row_generations=plan.row_generations,
    )
    with pytest.raises(ValueError, match="did not commit"):
        result.validate_against(plan, {"r0": [2, 3, 4, 5]})

    proven_result = ResidentEpochResult(
        **{
            **result.__dict__,
            "kv_imported": True,
            "kv_import_checksum": 123,
            "kv_snapshot_checksum": 123,
        }
    )
    proven_result.validate_against(plan, {"r0": [2, 3, 4, 5]})

    mismatched_result = ResidentEpochResult(
        **{
            **proven_result.__dict__,
            "kv_import_checksum": 124,
        }
    )
    with pytest.raises(ValueError, match="equivalent imported KV"):
        mismatched_result.validate_against(plan, {"r0": [2, 3, 4, 5]})

    invalid_request = ResidentEpochRequest(
        **{
            **importing_request.__dict__,
            "state_owner": "device",
        }
    )
    invalid_plan = ResidentEpochPlan(
        version=base.version,
        graph_batch_size=base.graph_batch_size,
        max_steps=base.max_steps,
        logical_capacity=base.logical_capacity,
        requests=(invalid_request,),
        active_mask=base.active_mask,
    )
    with pytest.raises(ValueError, match="Host-owned"):
        invalid_plan.validate()
