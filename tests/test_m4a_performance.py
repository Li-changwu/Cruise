import asyncio
import json
import signal
from pathlib import Path
from types import SimpleNamespace

from experiments.m4a_performance.run_benchmark import (
    BLOCKED_ORDER,
    _completion_request_body,
    _run_load,
    _scenario_metrics,
    _stop_server,
    compare_results,
    load_manifest,
    percentile,
    server_command,
)
from experiments.m4a_performance.analyze_profiles import _analyze_route
from experiments.m4a_performance.verify_results import verify
from vllm_ascend_resident_epoch.benchmark_metrics import (
    ResidentEpochBenchmarkMetrics,
    replay_event_journal,
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
    assert "if run_step compare" in script
    assert "if run_step verify" in script
    assert script.index("if run_step compare") < script.index("if run_step verify")
    assert script.index("if run_step verify") < script.rindex("result-integrity.log")
    assert "local mode=$1 runtime=" not in script
    assert '--run-label "${mode}-profile"' in script
    assert 'local runtime=${scratch}/p/${route_code}' in script
    assert "huggingface-cli" not in script
    assert "wget " not in script
    assert "curl " not in script


def test_profile_barrier_is_released_after_warmups(tmp_path, monkeypatch):
    manifest = load_manifest(WORKLOAD)
    manifest = SimpleNamespace(
        served_model_name=manifest.served_model_name,
        warmups=(),
        scenarios=(),
    )
    ready = tmp_path / "ready.json"
    start = tmp_path / "start"
    start.write_text("start\n", encoding="utf-8")
    process = SimpleNamespace(pid=123, poll=lambda: None)

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: Client())
    warmups, scenarios = asyncio.run(
        _run_load(
            "http://127.0.0.1:1",
            manifest,
            process,
            profile_ready_file=ready,
            profile_start_file=start,
        )
    )

    assert warmups == []
    assert scenarios == []
    ready_data = json.loads(ready.read_text(encoding="utf-8"))
    assert ready_data["api_server_pid"] == 123
    assert ready_data["runner_pid"] > 0


def test_profile_analyzer_reports_only_observed_ai_core_idle_gaps(tmp_path):
    profile_root = tmp_path / "profiles"
    route_root = profile_root / "graph" / "summary"
    evidence = tmp_path / "evidence"
    route_root.mkdir(parents=True)
    evidence.mkdir()
    (route_root / "task_time_0.csv").write_text(
        "Task Type,Task Start Time(us),Task Duration(us),Op Name\n"
        "AI_CORE,10,5,a\n"
        "AI_CORE,12,10,b\n"
        "AICPU,25,4,c\n"
        "AI_VECTOR_CORE,30,5,d\n",
        encoding="utf-8",
    )

    result = _analyze_route(profile_root, "graph", evidence)

    assert result["ai_core_tasks_observed"]
    assert result["ai_core_task_count"] == 3
    assert result["ai_core_idle_gap_us"] == {
        "count": 1,
        "min": 8.0,
        "mean": 8.0,
        "p50": 8.0,
        "p95": 8.0,
        "p99": 8.0,
        "max": 8.0,
    }


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
        Path(command[command.index("--generation-config") + 1]).name
        == "generation_config"
        for command in commands.values()
    )
    generation_configs = {
        command[command.index("--generation-config") + 1]
        for command in commands.values()
    }
    assert len(generation_configs) == 1
    generation_config = json.loads(
        (Path(next(iter(generation_configs))) / "generation_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert generation_config == {"eos_token_id": 151645}
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


def test_m4a_graceful_stop_does_not_signal_the_engine_process_group(monkeypatch):
    signals = []

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert timeout == 180
            self.returncode = 0

    monkeypatch.setattr(
        "experiments.m4a_performance.run_benchmark.os.kill",
        lambda pid, sig: signals.append((pid, sig)),
    )
    assert _stop_server(Process()) == 0
    assert signals == [(123, signal.SIGTERM)]


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

    replayed = replay_event_journal(output.with_name("metrics.events.jsonl"))
    assert replayed["counters"] == payload["counters"]
    assert replayed["epoch_steps"] == payload["epoch_steps"]
    assert replayed["rejections"] == payload["rejections"]


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
