from types import SimpleNamespace

import pytest

from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
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
        fallback_safe=True,
        feed_calls=0,
        fetch_calls=0,
    )
    result.validate_against(plan, {"r0": [3]})
