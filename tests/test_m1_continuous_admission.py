import json
from pathlib import Path

import pytest

from experiments.m1_continuous_admission.run_differential import (
    compare_results,
    load_scenario,
)


SCENARIO = (
    Path(__file__).parents[1]
    / "experiments"
    / "m1_continuous_admission"
    / "scenario.json"
)


def test_scenario_has_bounded_nontrivial_a_b_c_requests():
    requests = load_scenario(SCENARIO)

    assert [request.request_id for request in requests] == ["A", "B", "C"]
    assert [request.max_tokens for request in requests] == [7, 2, 2]
    assert all(
        len(request.prompt_token_ids) >= 2
        and len(request.prompt_token_ids) + request.max_tokens - 1 <= 8
        for request in requests
    )


def test_scenario_rejects_wrong_admission_boundary(tmp_path):
    payload = json.loads(SCENARIO.read_text(encoding="utf-8"))
    payload["requests"][2]["admit_after"] = "A:complete"
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid admission boundary"):
        load_scenario(path)


def test_comparison_requires_exact_scheduler_state(tmp_path):
    state = {
        "pass": True,
        "scenario": [{"request_id": "A"}],
        "tokens": {"A": [1, 2]},
        "terminal_finish_reasons": {"A": 1},
        "terminal_stop_reasons": {"A": None},
        "final_request_state": {
            "A": {
                "status": "FINISHED_LENGTH_CAPPED",
                "num_computed_tokens": 3,
                "num_output_tokens": 2,
            }
        },
    }
    baseline = tmp_path / "baseline.json"
    cruise = tmp_path / "cruise.json"
    baseline.write_text(json.dumps(state), encoding="utf-8")
    cruise.write_text(json.dumps(state), encoding="utf-8")
    assert compare_results(baseline, cruise)["pass"]

    changed = json.loads(json.dumps(state))
    changed["final_request_state"]["A"]["num_computed_tokens"] = 4
    cruise.write_text(json.dumps(changed), encoding="utf-8")
    comparison = compare_results(baseline, cruise)
    assert not comparison["pass"]
    assert not comparison["checks"]["same_final_request_state"]
