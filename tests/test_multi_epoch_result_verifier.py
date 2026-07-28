from copy import deepcopy
from pathlib import Path

import pytest

from verify_multi_epoch_cohort_result import (
    EXPECTED_CLASSES,
    EXPECTED_EPOCHS,
    EXPECTED_GATE,
    EXPECTED_SUPPORT,
    EXPECTED_TOKENS,
    validate_result,
)


def make_result(tmp_path: Path) -> dict:
    artifacts = {}
    for key in ("baseline_result", "native_server", "air", "tiling"):
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
    epochs = []
    for expected in EXPECTED_EPOCHS:
        requests = {
            request_id: {
                "row": values[0],
                "generation": values[1],
                "position": values[2],
                "token_id": values[3],
            }
            for request_id, values in expected["requests"].items()
        }
        epochs.append(
            {
                "name": expected["name"],
                "pass": True,
                "model_executed": True,
                "engine_core_step_wall_us": 1000,
                "feed_calls": 1,
                "fetch_calls": 1,
                "socket_send_calls": 1,
                "socket_receive_calls": 1,
                "model_calls": 2,
                "declared_input_bytes": 260,
                "declared_output_bytes": 368,
                "declared_total_bytes": 628,
                "python_cpu_us": 10,
                "native_cpu_us": 10,
                "sampled_tokens": expected["sampled_tokens"],
                "result_row_generations": expected["row_generations"],
                "plan": {
                    "max_steps": 2,
                    "active_mask": expected["active_mask"],
                    "row_generations": expected["row_generations"],
                    "requests": requests,
                },
                "checks": {"complete": True},
            }
        )
    final_state = {
        "A": {"status": "FINISHED_LENGTH_CAPPED", "num_computed_tokens": 6, "num_output_tokens": 6},
        "B": {"status": "FINISHED_LENGTH_CAPPED", "num_computed_tokens": 2, "num_output_tokens": 2},
        "C": {"status": "FINISHED_LENGTH_CAPPED", "num_computed_tokens": 2, "num_output_tokens": 2},
    }
    expected_tokens = {
        "A": EXPECTED_TOKENS,
        "B": EXPECTED_TOKENS[:2],
        "C": EXPECTED_TOKENS[:2],
    }
    return {
        "gate": EXPECTED_GATE,
        "schema_version": 1,
        "pass": True,
        "support_boundary": EXPECTED_SUPPORT,
        "resolved_classes": EXPECTED_CLASSES,
        "distributed_initialized_during_engine": True,
        "distributed_initialized_after_shutdown": False,
        "engine_core_init_wall_us": 1000,
        "warmup": {
            "status": 0,
            "model_calls": 1,
            "feed_calls": 1,
            "fetch_calls": 1,
            "wall_us": 100,
            "native_cpu_us": 10,
            "declared_input_bytes": 260,
            "declared_output_bytes": 368,
        },
        "checks": {"complete": True},
        "artifacts": artifacts,
        "trace": {
            "pass": True,
            "checks": {"complete": True},
            "observed_tokens": expected_tokens,
            "expected_tokens": expected_tokens,
            "epochs": epochs,
            "final_request_state": final_state,
            "cleanup": {"model_executed": False, "engine_outputs": []},
        },
    }


def test_accepts_complete_multi_epoch_evidence(tmp_path):
    summary = validate_result(make_result(tmp_path), require_artifacts=True)
    assert summary["pass"] is True
    assert summary["service_epochs"] == 3
    assert summary["service_feed_calls"] == 3
    assert summary["total_model_calls"] == 7


def test_rejects_generation_reuse(tmp_path):
    result = deepcopy(make_result(tmp_path))
    result["trace"]["epochs"][2]["plan"]["row_generations"] = [1, 2, 0, 0]
    with pytest.raises(ValueError, match="plan generations changed"):
        validate_result(result, require_artifacts=True)


def test_rejects_lazy_first_service_epoch(tmp_path):
    result = deepcopy(make_result(tmp_path))
    result["trace"]["epochs"][0]["engine_core_step_wall_us"] = 10_000_000
    with pytest.raises(ValueError, match="lazy-load latency"):
        validate_result(result, require_artifacts=True)
