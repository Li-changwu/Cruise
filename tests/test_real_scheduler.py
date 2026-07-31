from pathlib import Path

import pytest

from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend_resident_epoch.config import ResidentEpochConfig
from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
    ResidentEpochResult,
    attach_result,
    get_plan,
)
from vllm_ascend_resident_epoch.scheduler import ResidentEpochScheduler
from vllm_ascend_resident_epoch.worker import ResidentEpochWorker


MODEL_CONFIG = Path(__file__).parent / "fixtures" / "qwen2-7b-config"
EOS_TOKEN_ID = 151645


def make_scheduler(batch_size: int):
    scheduler = create_scheduler(
        model=str(MODEL_CONFIG),
        max_num_seqs=batch_size,
        max_num_batched_tokens=128,
        block_size=128,
        max_model_len=128,
        skip_tokenizer_init=True,
    )
    scheduler.__class__ = ResidentEpochScheduler
    scheduler._resident_epoch_config = ResidentEpochConfig()
    scheduler._resident_epoch_last_rejection = None
    return scheduler


def add_greedy_requests(scheduler, batch_size: int, max_tokens: int = 8):
    requests = create_requests(
        num_requests=batch_size,
        num_tokens=1,
        max_tokens=max_tokens,
        block_size=128,
    )
    for request in requests:
        request.sampling_params.temperature = 0.0
        request.sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
        scheduler.add_request(request)
    return requests


def make_prefill_request(
    request_id: str, prompt_token_ids: list[int], max_tokens: int
) -> Request:
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        client_index=0,
        prompt_token_ids=prompt_token_ids,
        sampling_params=params,
        pooling_params=None,
    )


def commit_device_epoch(scheduler, scheduler_output, tokens_by_req):
    plan = get_plan(scheduler_output)
    assert plan is not None
    req_ids = list(plan.req_ids)
    output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[tokens_by_req[req_id] for req_id in req_ids],
    )
    importing = any(request.kv_import_required for request in plan.requests)
    attach_result(
        output,
        ResidentEpochResult(
            version=CONTRACT_VERSION,
            route="device",
            status=0,
            model_calls=plan.max_steps,
            computed_steps={
                req_id: len(tokens_by_req[req_id]) for req_id in req_ids
            },
            row_generations=plan.row_generations,
            kv_imported=importing,
            kv_import_checksum=1 if importing else 0,
            kv_snapshot_checksum=1 if importing else 0,
        ),
    )
    scheduler.update_from_output(scheduler_output, output)


@pytest.mark.parametrize(
    ("batch_size", "graph_batch_size"),
    [(1, 4), (2, 4), (3, 4), (4, 4)],
)
def test_real_scheduler_emits_fixed_graph_plan_and_commits_epoch(
    batch_size, graph_batch_size
):
    scheduler = make_scheduler(batch_size)
    requests = add_greedy_requests(scheduler, batch_size)

    scheduler_output = scheduler.schedule()
    plan = get_plan(scheduler_output)
    assert plan is not None
    assert plan.graph_batch_size == graph_batch_size
    assert plan.max_steps == 8
    assert plan.active_mask == (1,) * batch_size + (0,) * (
        graph_batch_size - batch_size
    )
    assert all(value == 1 for value in scheduler_output.num_scheduled_tokens.values())
    assert all(len(request.scheduler_block_ids) == 1 for request in plan.requests)

    req_ids = list(plan.req_ids)
    sampled = [[100 + step for step in range(plan.max_steps)] for _ in req_ids]
    output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled,
    )
    attach_result(
        output,
        ResidentEpochResult(
            version=CONTRACT_VERSION,
            route="device",
            status=0,
            model_calls=plan.max_steps,
            computed_steps={req_id: plan.max_steps for req_id in req_ids},
            row_generations=plan.row_generations,
        ),
    )
    scheduler.update_from_output(scheduler_output, output)

    for request in requests:
        assert request.num_computed_tokens == 8
        assert request.num_tokens == 9
        assert request.num_output_tokens == 8
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_unsupported_sampling_stays_on_host_path_without_lookahead():
    scheduler = make_scheduler(1)
    requests = add_greedy_requests(scheduler, 1)
    requests[0].sampling_params.temperature = 0.8

    scheduler_output = scheduler.schedule()
    assert get_plan(scheduler_output) is None
    assert scheduler._resident_epoch_last_rejection == "non-greedy-temperature"
    assert scheduler.num_lookahead_tokens == 0
    block_ids = scheduler.kv_cache_manager.get_blocks("0").get_block_ids()
    assert len(block_ids) == 1
    assert len(block_ids[0]) == 1


def test_real_scheduler_to_dedicated_worker_control_path(monkeypatch):
    monkeypatch.setenv(
        "VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY",
        "vllm_ascend_resident_epoch.testing:create_test_engine",
    )
    scheduler = make_scheduler(4)
    requests = add_greedy_requests(scheduler, 4, max_tokens=4)
    scheduler_output = scheduler.schedule()
    plan = get_plan(scheduler_output)
    assert plan is not None

    worker = ResidentEpochWorker(
        vllm_config=scheduler.vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:1",
        is_driver_worker=True,
    )
    worker.init_device()
    worker.load_model()
    compilation_times = worker.compile_or_warm_up_model()
    assert compilation_times.language_model > 0
    assert worker.warmup_output is not None
    assert worker.warmup_output.model_calls == 1
    specs = worker.get_kv_cache_spec()
    assert len(specs) == 28
    assert worker.determine_available_memory() == 8 * 28 * 262144

    output = worker.execute_model(scheduler_output)
    scheduler.update_from_output(scheduler_output, output)
    worker.shutdown()

    for request in requests:
        assert request.num_computed_tokens == 4
        assert request.num_output_tokens == 4
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_host_prefill_transitions_to_device_owned_decode():
    scheduler = make_scheduler(1)
    request = create_requests(
        num_requests=1,
        num_tokens=3,
        max_tokens=4,
        block_size=128,
    )[0]
    request.sampling_params.temperature = 0.0
    request.sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    scheduler.add_request(request)

    prefill = scheduler.schedule()
    assert get_plan(prefill) is None
    assert scheduler._resident_epoch_last_rejection == "not-single-token-decode"
    prefill_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[101]],
    )
    scheduler.update_from_output(prefill, prefill_output)
    assert request.num_computed_tokens == 3
    assert request.num_output_tokens == 1

    first_decode = scheduler.schedule()
    plan = get_plan(first_decode)
    assert plan is not None
    assert plan.max_steps == 2
    assert plan.requests[0].position == 3
    assert plan.requests[0].sequence_length == 4
    assert plan.requests[0].token_id == 101
    assert plan.requests[0].state_owner == "host"
    assert plan.requests[0].kv_import_required

    decode_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[102, 103]],
    )
    attach_result(
        decode_output,
        ResidentEpochResult(
            version=CONTRACT_VERSION,
            route="device",
            status=0,
            model_calls=2,
            computed_steps={request.request_id: 2},
            row_generations=plan.row_generations,
            kv_imported=True,
            kv_import_checksum=1,
            kv_snapshot_checksum=1,
        ),
    )
    scheduler.update_from_output(first_decode, decode_output)
    assert request.request_id in scheduler._resident_epoch_device_owned

    next_decode = scheduler.schedule()
    next_plan = get_plan(next_decode)
    assert next_plan is not None
    assert next_plan.requests[0].state_owner == "device"
    assert not next_plan.requests[0].kv_import_required


def test_nontrivial_continuous_admission_isolates_host_prefill_and_reuses_row():
    scheduler = make_scheduler(2)
    scheduler._resident_epoch_config = ResidentEpochConfig(max_steps=2)
    request_a = make_prefill_request("A", [9707, 11], max_tokens=7)
    scheduler.add_request(request_a)

    prefill_a = scheduler.schedule()
    assert get_plan(prefill_a) is None
    scheduler.update_from_output(
        prefill_a,
        ModelRunnerOutput(
            req_ids=["A"],
            req_id_to_index={"A": 0},
            sampled_token_ids=[[101]],
        ),
    )
    import_a = scheduler.schedule()
    plan_a = get_plan(import_a)
    assert plan_a is not None
    assert plan_a.max_steps == 2
    commit_device_epoch(scheduler, import_a, {"A": [102, 103]})

    request_b = make_prefill_request("B", [9707, 11, 358], max_tokens=2)
    scheduler.add_request(request_b)
    prefill_b = scheduler.schedule()
    assert get_plan(prefill_b) is None
    assert prefill_b.num_scheduled_tokens == {"B": 3}
    assert scheduler._resident_epoch_last_rejection == "host-prefill-admission"
    assert [request.request_id for request in scheduler.running] == ["A", "B"]
    scheduler.update_from_output(
        prefill_b,
        ModelRunnerOutput(
            req_ids=["B"],
            req_id_to_index={"B": 0},
            sampled_token_ids=[[201]],
        ),
    )

    import_b = scheduler.schedule()
    plan_b = get_plan(import_b)
    assert plan_b is not None
    assert plan_b.max_steps == 1
    plan_b_rows = [
        (request.req_id, request.row, request.generation)
        for request in plan_b.requests
    ]
    assert plan_b_rows == [
        ("A", 0, 1),
        ("B", 1, 2),
    ]
    assert plan_b.requests[0].state_owner == "device"
    assert not plan_b.requests[0].kv_import_required
    assert plan_b.requests[1].state_owner == "host"
    assert plan_b.requests[1].kv_import_required
    commit_device_epoch(scheduler, import_b, {"A": [104], "B": [202]})
    assert request_b.status == RequestStatus.FINISHED_LENGTH_CAPPED

    request_c = make_prefill_request(
        "C", [9707, 11, 358, 374], max_tokens=2
    )
    scheduler.add_request(request_c)
    prefill_c = scheduler.schedule()
    assert get_plan(prefill_c) is None
    assert prefill_c.num_scheduled_tokens == {"C": 4}
    assert scheduler._resident_epoch_last_rejection == "host-prefill-admission"
    scheduler.update_from_output(
        prefill_c,
        ModelRunnerOutput(
            req_ids=["C"],
            req_id_to_index={"C": 0},
            sampled_token_ids=[[301]],
        ),
    )

    import_c = scheduler.schedule()
    plan_c = get_plan(import_c)
    assert plan_c is not None
    plan_c_rows = [
        (request.req_id, request.row, request.generation)
        for request in plan_c.requests
    ]
    assert plan_c_rows == [
        ("A", 0, 1),
        ("C", 1, 3),
    ]
    assert plan_c.requests[0].state_owner == "device"
    assert plan_c.requests[1].state_owner == "host"
    assert plan_c.requests[1].kv_import_required
    commit_device_epoch(scheduler, import_c, {"A": [105], "C": [302]})
    assert request_c.status == RequestStatus.FINISHED_LENGTH_CAPPED

    finish_a = scheduler.schedule()
    finish_plan = get_plan(finish_a)
    assert finish_plan is not None
    assert finish_plan.req_ids == ("A",)
    assert finish_plan.max_steps == 2
    commit_device_epoch(scheduler, finish_a, {"A": [106, 107]})
    assert request_a.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_ineligible_admission_uses_host_lane_without_advancing_device_request():
    scheduler = make_scheduler(2)
    scheduler._resident_epoch_config = ResidentEpochConfig(max_steps=2)
    request_a = make_prefill_request("A", [9707, 11], max_tokens=7)
    scheduler.add_request(request_a)
    prefill_a = scheduler.schedule()
    scheduler.update_from_output(
        prefill_a,
        ModelRunnerOutput(
            req_ids=["A"],
            req_id_to_index={"A": 0},
            sampled_token_ids=[[101]],
        ),
    )
    import_a = scheduler.schedule()
    commit_device_epoch(scheduler, import_a, {"A": [102, 103]})

    unsupported = make_prefill_request("unsupported", [9707, 11], max_tokens=2)
    unsupported.sampling_params.temperature = 0.8
    scheduler.add_request(unsupported)

    unsupported_prefill = scheduler.schedule()
    assert get_plan(unsupported_prefill) is None
    assert unsupported_prefill.num_scheduled_tokens == {"unsupported": 2}
    assert scheduler._resident_epoch_last_rejection == "host-prefill-admission"
    scheduler.update_from_output(
        unsupported_prefill,
        ModelRunnerOutput(
            req_ids=["unsupported"],
            req_id_to_index={"unsupported": 0},
            sampled_token_ids=[[201]],
        ),
    )

    unsupported_decode = scheduler.schedule()
    assert get_plan(unsupported_decode) is None
    assert unsupported_decode.num_scheduled_tokens == {"unsupported": 1}
    assert scheduler._resident_epoch_last_rejection == "unsupported-host-isolation"
    scheduler.update_from_output(
        unsupported_decode,
        ModelRunnerOutput(
            req_ids=["unsupported"],
            req_id_to_index={"unsupported": 0},
            sampled_token_ids=[[202]],
        ),
    )
    assert unsupported.status == RequestStatus.FINISHED_LENGTH_CAPPED

    resume_a = scheduler.schedule()
    resume_plan = get_plan(resume_a)
    assert resume_plan is not None
    assert resume_plan.req_ids == ("A",)
    assert resume_plan.requests[0].state_owner == "device"
    assert resume_plan.requests[0].generation == 1


def test_two_simultaneous_prefills_are_isolated_from_device_owned_request():
    scheduler = make_scheduler(3)
    scheduler._resident_epoch_config = ResidentEpochConfig(max_steps=2)
    request_a = make_prefill_request("A", [9707, 11], max_tokens=7)
    scheduler.add_request(request_a)
    prefill_a = scheduler.schedule()
    scheduler.update_from_output(
        prefill_a,
        ModelRunnerOutput(
            req_ids=["A"],
            req_id_to_index={"A": 0},
            sampled_token_ids=[[101]],
        ),
    )
    import_a = scheduler.schedule()
    commit_device_epoch(scheduler, import_a, {"A": [102, 103]})

    request_b = make_prefill_request("B", [9707, 11, 358], max_tokens=3)
    request_c = make_prefill_request("C", [9707, 11, 358, 374], max_tokens=3)
    scheduler.add_request(request_b)
    scheduler.add_request(request_c)

    prefill_bc = scheduler.schedule()
    assert get_plan(prefill_bc) is None
    assert prefill_bc.num_scheduled_tokens == {"B": 3, "C": 4}
    assert scheduler._resident_epoch_last_rejection == "host-prefill-admission"
    scheduler.update_from_output(
        prefill_bc,
        ModelRunnerOutput(
            req_ids=["B", "C"],
            req_id_to_index={"B": 0, "C": 1},
            sampled_token_ids=[[201], [301]],
        ),
    )

    import_bc = scheduler.schedule()
    plan_bc = get_plan(import_bc)
    assert plan_bc is not None
    assert plan_bc.req_ids == ("A", "B", "C")
    assert [request.state_owner for request in plan_bc.requests] == [
        "device",
        "host",
        "host",
    ]
    assert [request.kv_import_required for request in plan_bc.requests] == [
        False,
        True,
        True,
    ]


def test_delta_output_kind_allows_bounded_multi_token_epoch():
    scheduler = make_scheduler(2)
    scheduler._resident_epoch_config = ResidentEpochConfig(max_steps=4)
    requests = add_greedy_requests(scheduler, 2, max_tokens=6)
    requests[1].sampling_params.output_kind = RequestOutputKind.DELTA

    scheduler_output = scheduler.schedule()
    plan = get_plan(scheduler_output)

    assert plan is not None
    assert plan.max_steps == 4


def test_running_prefill_is_isolated_from_device_owned_requests():
    scheduler = make_scheduler(2)
    device_request = add_greedy_requests(scheduler, 1, max_tokens=6)[0]
    prefill_request = make_prefill_request("prefill", [9707, 11, 358], 4)
    prefill_request.num_computed_tokens = 1
    scheduler.running = [device_request, prefill_request]

    reason = scheduler._host_isolation_reason((device_request,))

    assert reason == "host-prefill-admission"
