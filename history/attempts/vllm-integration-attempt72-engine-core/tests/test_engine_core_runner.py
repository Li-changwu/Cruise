from pathlib import Path

import torch

from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from run_engine_core_native import make_engine_vllm_config, run_case
from vllm_ascend_resident_epoch.scheduler import ResidentEpochScheduler
from vllm_ascend_resident_epoch.worker import ResidentEpochWorker


MODEL_CONFIG = Path(__file__).parent / "fixtures" / "qwen2-7b-config"


def test_engine_core_resolves_resident_scheduler_worker_and_epoch(monkeypatch):
    monkeypatch.setenv(
        "VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY",
        "vllm_ascend_resident_epoch.testing:create_test_engine",
    )
    config = make_engine_vllm_config(MODEL_CONFIG)
    engine_core = EngineCore(
        vllm_config=config,
        executor_class=UniProcExecutor,
        log_stats=True,
    )
    try:
        assert isinstance(engine_core.scheduler, ResidentEpochScheduler)
        assert isinstance(engine_core.model_executor, UniProcExecutor)
        assert isinstance(
            engine_core.model_executor.driver_worker.worker,
            ResidentEpochWorker,
        )
        assert torch.distributed.is_initialized()

        case = run_case(
            engine_core=engine_core,
            name="engine-core-test",
            batch_size=2,
            max_steps=4,
            expected_tokens=[11691, 11692, 11693, 11694],
            eos_token_ids=[151645, 151645],
            input_token_id=11690,
        )
        failed_checks = {
            key: value for key, value in case["checks"].items() if not value
        }
        assert not failed_checks, failed_checks
        assert case["feed_calls"] == 1
        assert case["fetch_calls"] == 1
    finally:
        engine_core.shutdown()

    assert not torch.distributed.is_initialized()
