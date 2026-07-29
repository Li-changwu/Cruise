import json
from pathlib import Path

import pytest

from experiments.m1_batched_prefill.run_differential import (
    IMPORT_INPUT_BYTES,
    OUTPUT_BYTES,
    STEADY_INPUT_BYTES,
    _case_checks,
    compare_results,
    load_case_manifest,
)


CASES = (
    Path(__file__).parents[1]
    / "experiments"
    / "m1_batched_prefill"
    / "cases.json"
)


def test_manifest_covers_b1_through_b4_with_bounded_nontrivial_prompts():
    cases = load_case_manifest(CASES)

    assert [case.batch_size for case in cases] == [1, 2, 3, 4]
    assert all(
        len(request.prompt_token_ids) >= 2
        and len(request.prompt_token_ids) + request.max_tokens - 1 <= 8
        for case in cases
        for request in case.requests
    )
    assert all(
        len({request.max_tokens for request in case.requests}) > 1
        for case in cases
        if case.batch_size > 1
    )


def test_manifest_rejects_request_outside_resident_capacity(tmp_path):
    payload = json.loads(CASES.read_text(encoding="utf-8"))
    payload["cases"][0]["requests"][0]["max_tokens"] = 7
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exceed resident capacity"):
        load_case_manifest(path)


def test_case_checks_accept_mixed_batch_import_then_device_owned_shrink():
    spec = load_case_manifest(CASES)[1]
    req0, req1 = (request.request_id for request in spec.requests)

    def plan(requests, *, importing):
        return {
            "max_steps": 2,
            "active_mask": [1 if row < len(requests) else 0 for row in range(4)],
            "row_generations": [1, 2, 0, 0],
            "requests": [
                {
                    "req_id": req_id,
                    "row": row,
                    "generation": row + 1,
                    "state_owner": "host" if importing else "device",
                    "kv_import_required": importing,
                }
                for row, req_id in enumerate(requests)
            ],
        }

    def result(requests, *, importing):
        return {
            "route": "device",
            "status": 0,
            "commit_state": "COMMITTED",
            "computed_steps": {req_id: 2 for req_id in requests},
            "row_generations": [1, 2, 0, 0],
            "feed_calls": 1,
            "fetch_calls": 1,
            "declared_input_bytes": (
                IMPORT_INPUT_BYTES if importing else STEADY_INPUT_BYTES
            ),
            "declared_output_bytes": OUTPUT_BYTES,
            "kv_imported": importing,
            "host_kv_checksum": 123 if importing else 0,
            "device_kv_checksum": 123 if importing else 0,
        }

    case = {
        "tokens": {req0: [1, 2, 3, 4, 5], req1: [6, 7, 8]},
        "request_drained": True,
        "final_request_state": {
            req0: {"status": "FINISHED_LENGTH_CAPPED"},
            req1: {"status": "FINISHED_LENGTH_CAPPED"},
        },
        "steps": [
            {
                "model_executed": True,
                "new_tokens_by_request": {req0: [1], req1: [6]},
                "engine_outputs": [{"request_id": req0}, {"request_id": req1}],
                "plan": None,
                "result": None,
            },
            {
                "model_executed": True,
                "new_tokens_by_request": {req0: [2, 3], req1: [7, 8]},
                "engine_outputs": [{"request_id": req0}, {"request_id": req1}],
                "plan": plan([req0, req1], importing=True),
                "result": result([req0, req1], importing=True),
            },
            {
                "model_executed": True,
                "new_tokens_by_request": {req0: [4, 5]},
                "engine_outputs": [{"request_id": req0}],
                "plan": {
                    **plan([req0], importing=False),
                    "active_mask": [1, 0, 0, 0],
                },
                "result": {
                    **result([req0], importing=False),
                    "row_generations": [1, 0, 0, 0],
                },
            },
            {
                "model_executed": False,
                "new_tokens_by_request": {},
                "engine_outputs": [],
                "plan": None,
                "result": None,
            },
        ],
    }
    case["steps"][2]["plan"]["row_generations"] = [1, 0, 0, 0]

    checks = _case_checks(spec, case, "cruise")

    assert checks
    assert all(checks.values())


def test_compare_requires_exact_per_request_outputs(tmp_path):
    common_case = {
        "name": "b1",
        "tokens": {"b1-r0": [1, 2]},
        "terminal_finish_reasons": {"b1-r0": 1},
        "terminal_stop_reasons": {"b1-r0": None},
        "final_request_state": {
            "b1-r0": {
                "status": "FINISHED_LENGTH_CAPPED",
                "num_computed_tokens": 4,
                "num_output_tokens": 2,
            }
        },
    }
    baseline = {
        "pass": True,
        "manifest": [{"name": "b1"}],
        "cases": [common_case],
    }
    cruise = json.loads(json.dumps(baseline))
    baseline_path = tmp_path / "baseline.json"
    cruise_path = tmp_path / "cruise.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    cruise_path.write_text(json.dumps(cruise), encoding="utf-8")

    assert compare_results(baseline_path, cruise_path)["pass"]

    cruise["cases"][0]["tokens"]["b1-r0"][-1] = 3
    cruise_path.write_text(json.dumps(cruise), encoding="utf-8")
    comparison = compare_results(baseline_path, cruise_path)
    assert not comparison["pass"]
    assert not comparison["case_checks"]["b1"]["same_tokens"]
