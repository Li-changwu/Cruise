import json
from pathlib import Path

import pytest

from experiments.m1_batched_prefill.run_differential import (
    DEFAULT_KV_CACHE_BYTES,
    configured_kv_cache_bytes,
)
from experiments.m1_exit.run_differential import (
    REQUEST_KINDS,
    compare_results,
    load_workload,
)


WORKLOAD = (
    Path(__file__).parents[1] / "experiments" / "m1_exit" / "workload.json"
)


def test_exit_workload_has_exact_coverage():
    workload = load_workload(WORKLOAD)
    summary = workload.summary()

    assert summary["cohort_count"] == 400
    assert summary["request_count"] == 1000
    assert summary["batch_size_counts"] == {1: 100, 2: 100, 3: 100, 4: 100}
    assert set(summary["kind_counts"]) == set(REQUEST_KINDS)
    assert set(summary["prompt_length_counts"]) == {2, 3, 4, 5}
    assert set(summary["output_budget_counts"]) >= {2, 3, 4}
    assert all(
        len(request.prompt_token_ids) + request.max_tokens - 1 <= 8
        for request in workload.requests
    )


def test_exit_workload_is_deterministic():
    first = load_workload(WORKLOAD)
    second = load_workload(WORKLOAD)

    assert first.digest() == second.digest()
    assert [request.as_record() for request in first.requests] == [
        request.as_record() for request in second.requests
    ]


def test_m1_kv_cache_budget_is_explicit_and_validated(monkeypatch):
    monkeypatch.delenv("CRUISE_VLLM_KV_CACHE_BYTES", raising=False)
    assert configured_kv_cache_bytes() == DEFAULT_KV_CACHE_BYTES

    monkeypatch.setenv("CRUISE_VLLM_KV_CACHE_BYTES", str(768 * 1024 * 1024))
    assert configured_kv_cache_bytes() == 768 * 1024 * 1024

    monkeypatch.setenv("CRUISE_VLLM_KV_CACHE_BYTES", "1024")
    with pytest.raises(ValueError, match="smaller than"):
        configured_kv_cache_bytes()


def test_exit_comparison_reports_exact_field_mismatch(tmp_path):
    state = {
        "pass": True,
        "workload_sha256": "fixed",
        "workload_summary": {"request_count": 1000},
        "cases": [
            {
                "name": f"case-{index:03d}",
                "tokens": {"request": [1, 2]},
                "terminal_finish_reasons": {"request": 1},
                "terminal_stop_reasons": {"request": None},
                "cancelled_at": {},
                "final_request_state": {"request": {"status": "FINISHED"}},
            }
            for index in range(400)
        ],
    }
    baseline = tmp_path / "baseline.json"
    cruise = tmp_path / "cruise.json"
    baseline.write_text(json.dumps(state), encoding="utf-8")
    cruise.write_text(json.dumps(state), encoding="utf-8")
    assert compare_results(baseline, cruise)["pass"]

    changed = json.loads(json.dumps(state))
    changed["cases"][17]["tokens"]["request"] = [1, 3]
    cruise.write_text(json.dumps(changed), encoding="utf-8")
    result = compare_results(baseline, cruise)
    assert not result["pass"]
    assert result["mismatches"] == {"case-017": ["tokens"]}
