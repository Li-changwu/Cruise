import json
from pathlib import Path
from types import SimpleNamespace

from experiments.m4a_performance.run_benchmark import (
    BLOCKED_ORDER,
    _completion_request_body,
    _scenario_metrics,
    compare_results,
    load_manifest,
    percentile,
    server_command,
)
from experiments.m4a_performance.verify_results import verify
from vllm_ascend_resident_epoch.benchmark_metrics import (
    ResidentEpochBenchmarkMetrics,
)


ROOT = Path(__file__).parents[1]
WORKLOAD = ROOT / "experiments" / "m4a_performance" / "workload.json"
RUNNER = ROOT / "experiments" / "m4a_performance" / "run_on_910b2.sh"


def test_m4a_manifest_freezes_primary_negative_regime_and_thresholds():
    manifest = load_manifest(WORKLOAD)
    scenarios = {scenario.name: scenario for scenario in manifest.scenarios}

    assert manifest.primary_scenario == "decode-stream-c4"
    assert scenarios[manifest.primary_scenario].stream
    assert scenarios[manifest.primary_scenario].concurrency == 4
    assert scenarios["decode-nonstream-c4"].stream is False
    assert scenarios["decode-overload-c8"].concurrency == 8
    assert manifest.thresholds == {
        "median_tpot_improvement_percent": 15.0,
        "p95_tpot_improvement_percent": 15.0,
        "host_cpu_per_token_reduction_percent": 30.0,
    }


def test_m4a_percentile_uses_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 95.0) == 3.85


def test_m4a_hardware_runner_preserves_storage_and_milestone_contract():
    script = RUNNER.read_text(encoding="utf-8")

    assert "storage_guard_preflight" in script
    assert "storage_guard_cleanup_scratch" in script
    assert "runtime_weight_digest=2ec95bf8e78cfaf091782b3c531b19b9cced35dcfab0e418c756e25abe456761" in script
    assert "order=(eager-1 graph-1 cruise-1 cruise-2 graph-2 eager-2 graph-3 cruise-3 eager-3)" in script
    assert "formal_m2\\topen" in script
    assert "formal_m3\\topen" in script
    assert "formal_m4\\topen" in script
    assert "huggingface-cli" not in script
    assert "wget " not in script
    assert "curl " not in script


def test_m4a_commands_separate_eager_graph_and_cruise(tmp_path):
    manifest = load_manifest(WORKLOAD)
    commands = {
        mode: server_command(
            mode=mode,
            model=tmp_path / "model",
            manifest=manifest,
            port=8000,
        )
        for mode in ("eager", "graph", "cruise")
    }

    assert "--enforce-eager" in commands["eager"]
    assert "--enforce-eager" not in commands["graph"]
    assert "--scheduler-cls" not in commands["graph"]
    assert "--scheduler-cls" in commands["cruise"]
    assert all("--no-async-scheduling" in command for command in commands.values())
    assert all(
        command[command.index("--generation-config") + 1] == "vllm"
        for command in commands.values()
    )
    budgets = {
        command[command.index("--kv-cache-memory-bytes") + 1]
        for command in commands.values()
    }
    assert budgets == {str(512 * 1024 * 1024)}


def test_m4a_requests_explicitly_freeze_supported_greedy_sampling():
    manifest = load_manifest(WORKLOAD)
    body = _completion_request_body("cruise-m4a", manifest.scenarios[0])

    assert body == {
        "model": "cruise-m4a",
        "prompt": [9707, 11],
        "max_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_tokens": 0,
        "ignore_eos": False,
        "return_token_ids": True,
    }


def test_benchmark_metrics_are_disabled_by_default_and_flush_once(tmp_path):
    disabled = ResidentEpochBenchmarkMetrics()
    disabled.record_schedule(SimpleNamespace(requests=(1,), max_steps=2), None)
    assert disabled.as_record()["counters"] == {}

    output = tmp_path / "metrics.json"
    metrics = ResidentEpochBenchmarkMetrics(output)
    metrics.record_schedule(SimpleNamespace(requests=(1, 2), max_steps=2), None)
    metrics.record_schedule(None, "host-prefill-in-progress")
    metrics.record_result(
        SimpleNamespace(
            route="device",
            model_calls=2,
            computed_steps={"a": 2, "b": 2},
            feed_calls=1,
            fetch_calls=1,
            wall_us=100,
            native_cpu_us=20,
            socket_send_calls=1,
            socket_receive_calls=1,
            kv_imported=True,
        )
    )
    metrics.flush()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["counters"]["device_epochs"] == 1
    assert payload["counters"]["device_request_tokens"] == 4
    assert payload["counters"]["host_schedule_calls"] == 1
    assert payload["epoch_steps"] == {"2": 1}
    assert payload["rejections"] == {"host-prefill-in-progress": 1}


def _result(mode, label, manifest, *, tpot_ms, cpu_ms_per_token):
    scenarios = []
    for scenario in manifest.scenarios:
        tokens = list(range(scenario.max_tokens))
        record = {
            "request_index": 0,
            "tokens": tokens,
            "finish_reason": "length",
            "stop_reason": None,
            "done": True,
            "latency_ms": tpot_ms * scenario.max_tokens,
            "ttft_ms": 10.0 if scenario.stream else None,
            "tpot_ms": tpot_ms if scenario.stream else None,
            "inter_token_ms": (
                [tpot_ms] * (scenario.max_tokens - 1) if scenario.stream else []
            ),
            "pass": True,
        }
        cpu_seconds = cpu_ms_per_token * len(tokens) / 1000.0
        scenarios.append(
            {
                "scenario": scenario.as_record(),
                "records": [record],
                "metrics": _scenario_metrics(
                    [record], duration_ms=1000.0, cpu_seconds=cpu_seconds
                ),
                "process_tree_before": {"cpu_seconds": 1.0},
                "process_tree_after": {"cpu_seconds": 1.0 + cpu_seconds},
                "pass": True,
            }
        )
    return {
        "schema_version": 1,
        "mode": mode,
        "run_label": label,
        "scenarios": scenarios,
        "resident_route_metrics": (
            {
                "counters": {
                    "device_request_tokens": manifest.expected_device_request_tokens()
                }
            }
            if mode == "cruise"
            else None
        ),
        "pass": True,
    }


def _write_results(tmp_path, *, cruise_tpot):
    manifest = load_manifest(WORKLOAD)
    settings = {
        "eager": (100.0, 1.0),
        "graph": (80.0, 0.8),
        "cruise": (cruise_tpot, 0.4),
    }
    paths = []
    for label in BLOCKED_ORDER:
        mode = label.split("-", 1)[0]
        tpot, cpu = settings[mode]
        path = tmp_path / f"{label}.json"
        path.write_text(
            json.dumps(
                _result(
                    mode,
                    label,
                    manifest,
                    tpot_ms=tpot,
                    cpu_ms_per_token=cpu,
                )
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return manifest, paths


def test_m4a_comparison_uses_one_strongest_baseline_and_passes_threshold(tmp_path):
    manifest, paths = _write_results(tmp_path, cruise_tpot=60.0)
    comparison = compare_results(paths, manifest)

    assert comparison["execution_pass"]
    assert comparison["qualification_pass"]
    assert comparison["threshold_values"]["strongest_baseline"] == "graph"
    assert comparison["threshold_values"]["median_tpot_improvement_percent"] == 25.0
    assert comparison["formal_milestones_closed"] == []


def test_threshold_failure_is_retained_as_valid_m4a_execution(tmp_path):
    manifest, paths = _write_results(tmp_path, cruise_tpot=90.0)
    comparison = compare_results(paths, manifest)

    assert comparison["execution_pass"]
    assert not comparison["qualification_pass"]
    assert comparison["pass"]
    assert comparison["decision"] == "performance-attribution-required"


def test_independent_verifier_reconstructs_primary_metrics(tmp_path):
    manifest, paths = _write_results(tmp_path, cruise_tpot=60.0)
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(compare_results(paths, manifest)), encoding="utf-8"
    )

    result = verify(comparison_path, WORKLOAD, paths)

    assert result["pass"]
    assert result["execution_pass"]
    assert result["qualification_pass"]
    assert result["checks"]["reported_values_match"]
