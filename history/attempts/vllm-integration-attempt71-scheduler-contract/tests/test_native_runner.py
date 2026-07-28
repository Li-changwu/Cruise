import json
from pathlib import Path

from vllm_ascend_resident_epoch.contract import get_plan

from run_scheduler_native import (
    load_expected_tokens,
    make_request,
    make_scheduler,
    make_vllm_config,
)


MODEL_CONFIG = Path(__file__).parent / "fixtures" / "qwen2-7b-config"


def test_native_runner_constructs_resident_scheduler_directly():
    vllm_config = make_vllm_config(MODEL_CONFIG)
    scheduler = make_scheduler(vllm_config)
    request = make_request(
        request_id="native-r0",
        client_index=0,
        max_tokens=4,
        eos_token_id=151645,
        input_token_id=11690,
    )
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    plan = get_plan(scheduler_output)

    assert plan is not None
    assert plan.graph_batch_size == 4
    assert plan.max_steps == 4
    assert plan.requests[0].token_id == 11690
    assert plan.requests[0].scheduler_block_ids


def test_native_runner_extracts_frozen_token_oracle(tmp_path):
    cases = []
    for max_steps in (1, 2, 4, 8):
        cases.append(
            {
                "case": f"k{max_steps}-baseline",
                "max_steps": max_steps,
                "steps": [
                    {
                        "request": 0,
                        "step": step + 1,
                        "device_token": 100 + step,
                    }
                    for step in range(max_steps)
                ],
            }
        )
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({"pass": True, "cases": cases}),
        encoding="utf-8",
    )

    expected = load_expected_tokens(path)

    assert expected == {
        1: [100],
        2: [100, 101],
        4: [100, 101, 102, 103],
        8: [100, 101, 102, 103, 104, 105, 106, 107],
    }
