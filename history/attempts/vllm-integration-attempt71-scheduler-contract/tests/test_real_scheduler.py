from pathlib import Path

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

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
