from typing import Any


def request_rejection_reason(request: Any) -> str | None:
    if request.pooling_params is not None:
        return "pooling-request"
    if request.mm_features:
        return "multimodal-input"
    if request.lora_request is not None:
        return "lora"
    if request.prompt_embeds is not None:
        return "prompt-embeds"
    if request.use_structured_output:
        return "structured-output"

    params = request.sampling_params
    if params is None:
        return "missing-sampling-params"
    if params.temperature != 0.0:
        return "non-greedy-temperature"
    if params.top_p != 1.0 or params.top_k != 0 or params.min_p != 0.0:
        return "unsupported-sampling-filter"
    if (
        params.presence_penalty != 0.0
        or params.frequency_penalty != 0.0
        or params.repetition_penalty != 1.0
    ):
        return "sampling-penalty"
    if params.min_tokens != 0:
        return "min-tokens"
    if params.ignore_eos:
        return "ignore-eos"
    if params.eos_token_id is None:
        return "missing-eos"
    if params.stop or params.stop_token_ids:
        return "extra-stop-condition"
    if params.logprobs is not None or params.prompt_logprobs is not None:
        return "logprobs"
    if params.logit_bias or params.allowed_token_ids:
        return "logit-processor"
    if params.bad_words:
        return "bad-words"
    if params.repetition_detection is not None:
        return "repetition-detection"
    if params.extra_args:
        return "extra-sampling-args"
    return None

