from copy import deepcopy
from pathlib import Path

import pytest

from verify_engine_core_result import (
    EXPECTED_CLASSES,
    EXPECTED_GATE,
    EXPECTED_SUPPORT,
    REQUIRED_CHECKS,
    expected_cases,
    validate_result,
)


def make_result(tmp_path: Path) -> dict:
    artifacts = {}
    for key in ("baseline_result", "native_library", "native_server", "air", "tiling"):
        path = tmp_path / key
        path.touch()
        artifacts[key] = str(path)
    weights = tmp_path / "external_weights"
    weights.mkdir()
    artifacts.update(
        {
            "external_weights": str(weights),
            "baseline_sha256": "a" * 64,
            "backend_factory": (
                "vllm_ascend_resident_epoch.sidecar_backend:create_sidecar_engine"
            ),
        }
    )

    cases = []
    for name, (batch_size, max_steps) in expected_cases().items():
        steps = [1, 4, 1, 4] if name == "b4-independent-eos" else [max_steps] * batch_size
        computed = {f"{name}-r{row}": step for row, step in enumerate(steps)}
        cases.append(
            {
                "name": name,
                "pass": True,
                "batch_size": batch_size,
                "max_steps": max_steps,
                "feed_calls": 1,
                "fetch_calls": 1,
                "model_calls": max(steps),
                "computed_steps": computed,
                "cleanup_engine_outputs": [],
                "engine_outputs": [{} for _ in range(batch_size)],
                "plan": {
                    "graph_batch_size": 4,
                    "active_mask": [1] * batch_size + [0] * (4 - batch_size),
                },
                "checks": {key: True for key in REQUIRED_CHECKS},
            }
        )
    return {
        "gate": EXPECTED_GATE,
        "schema_version": 1,
        "pass": True,
        "support_boundary": EXPECTED_SUPPORT,
        "resolved_classes": EXPECTED_CLASSES,
        "distributed_initialized_during_engine": True,
        "distributed_initialized_after_shutdown": False,
        "artifacts": artifacts,
        "cases": cases,
    }


def test_accepts_complete_engine_core_evidence(tmp_path):
    summary = validate_result(make_result(tmp_path), require_artifacts=True)
    assert summary["pass"] is True
    assert summary["case_count"] == 13
    assert summary["feed_calls"] == 13
    assert summary["fetch_calls"] == 13


def test_rejects_incomplete_case_matrix(tmp_path):
    result = make_result(tmp_path)
    result["cases"].pop()
    with pytest.raises(ValueError, match="exactly 13"):
        validate_result(result, require_artifacts=True)


def test_rejects_model_execution_during_cleanup(tmp_path):
    result = deepcopy(make_result(tmp_path))
    result["cases"][0]["checks"]["cleanup_model_not_executed"] = False
    with pytest.raises(ValueError, match="checks failed"):
        validate_result(result, require_artifacts=True)
