from __future__ import annotations

import ast
import asyncio
import json
import os
from pathlib import Path
import time
from types import ModuleType

from fastapi.testclient import TestClient
import pytest

from experiments.persistent_decode_p5.run_p5_benchmark import (
    GRAPH_ASCEND_ADDITIONAL_CONFIG,
    GRAPH_PLUGIN_ALLOWLIST,
    GRAPH_RMSNORM_BACKEND,
    ProcessTreeCpuMeter,
    OWNER_STARTUP_TIMEOUT_SECONDS,
    SERVICE_READINESS_TIMEOUT_SECONDS,
    _write_diagnostic_tail,
    START_ORDER,
    _verified_revision,
    load_workload,
)
from experiments.persistent_decode_p5.verify_p5_gate import verify
from experiments.persistent_decode_p5.verify_p5_pair import verify_pair
from vllm_ascend_persistent_owner.owner_transport import (
    OwnerEvent,
    OwnerRequest,
    OwnerTransport,
    _parse_fields,
)
from vllm_ascend_persistent_owner.graph_rmsnorm_compat import (
    BACKEND as COMPAT_RMSNORM_BACKEND,
    install_cann9_rmsnorm_compat,
)
from vllm_ascend_persistent_owner.server import (
    ServiceSettings,
    _validate_request,
    create_app,
)


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "experiments" / "persistent_decode_p5"
OWNER_PACKAGE = ROOT / "src" / "vllm_ascend_persistent_owner"


def test_p5_workload_freezes_formal_matrix() -> None:
    workload = load_workload(P5 / "workload.json")
    assert workload.start_order == START_ORDER
    assert workload.model_manifest_sha256 == (
        "651d64436c415afd6faf3d82f14086c2df54d99c02f66f81a15b83a2d17de5f9"
    )
    assert workload.primary == {
        "prompt_token_ids": list(range(1000, 1128)),
        "output_tokens": 256,
        "request_count": 32,
        "concurrency": 4,
        "streaming": True,
        "ignore_eos": True,
    }
    assert workload.thresholds == {
        "host_cpu_per_token_reduction_percent": 50.0,
        "tpot_p50_improvement_percent": 15.0,
        "tpot_p95_improvement_percent": 15.0,
        "output_throughput_improvement_percent": 15.0,
    }


def test_p5_driver_requires_full_model_revision_and_sha256_manifest() -> None:
    driver = (P5 / "run_on_910b.sh").read_text(encoding="utf-8")
    assert "a09a35458c702b33eeacc393d103063234e8bc28" in driver
    assert ".cruise-model-revision" in driver
    assert ".cruise-model-sha256" in driver
    assert "sha256sum --check --strict" in driver
    assert "CRUISE_P5_STOP_AFTER_GRAPH1" in driver
    assert "P5_GRAPH1_SMOKE_COMPLETE" in driver
    assert "CRUISE_P5_STOP_AFTER_FIRST_PAIR" in driver
    assert "P5_FIRST_PAIR_COMPLETE" in driver
    assert (
        "STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-5" in driver
    )


def test_owner_entry_has_no_legacy_runtime_import() -> None:
    forbidden = {
        "vllm_ascend_resident_epoch.scheduler",
        "vllm_ascend_resident_epoch.sidecar_backend",
        "vllm_ascend_resident_epoch.server_launcher",
    }
    imported: set[str] = set()
    for path in OWNER_PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert imported.isdisjoint(forbidden)
    native = (P5 / "persistent_owner_transport.cpp").read_text(encoding="utf-8")
    assert "FeedEvent(session, event" in native
    assert "P5_OWNER_READY" in native
    assert "phase=compile_start" in native
    assert "phase=compile_returned" in native
    assert "phase=drain_started" in native
    assert "kEventAdmit" in native
    assert "resident_epoch_server" not in native
    benchmark = (P5 / "run_p5_benchmark.py").read_text(encoding="utf-8")
    assert "vllm_ascend_resident_epoch" not in benchmark
    assert "vllm_ascend_persistent_owner.graph_baseline_launcher" in benchmark
    assert "ascend" in GRAPH_PLUGIN_ALLOWLIST
    assert "ascend_model" in GRAPH_PLUGIN_ALLOWLIST
    assert all("resident_epoch" not in name for name in GRAPH_PLUGIN_ALLOWLIST)
    assert json.loads(GRAPH_ASCEND_ADDITIONAL_CONFIG) == {
        "ascend_compilation_config": {"fuse_norm_quant": False}
    }
    assert GRAPH_RMSNORM_BACKEND == COMPAT_RMSNORM_BACKEND


def test_graph_rmsnorm_compat_is_local_and_idempotent() -> None:
    layernorm = ModuleType("fake_vllm_ascend_layernorm")
    layernorm.enable_custom_op = lambda: True  # type: ignore[attr-defined]
    assert install_cann9_rmsnorm_compat(layernorm) is True
    assert layernorm.enable_custom_op() is False  # type: ignore[attr-defined]
    assert install_cann9_rmsnorm_compat(layernorm) is False


def test_owner_output_protocol_updates_request_and_counters() -> None:
    prefix, fields = _parse_fields(
        "P4_OUTPUT type=2 event_seq=1 request=7 generation=9 row=2 "
        "commit_seq=1 token=42 position=128 page=1 status=0 aicore_calls=129 "
        "total_commits=1 total_retired=0 checksum=77 finish_reason=0"
    )
    assert prefix == "P4_OUTPUT"
    transport = OwnerTransport(("/bin/false",))
    request = OwnerRequest(7, 9, (1000,), 1)
    transport.requests[request.key] = request
    transport._dispatch_output(fields)
    event = request.events.get_nowait()
    assert event.kind == "commit"
    assert event.token == 42
    assert event.page == 1
    assert transport.metrics()["counters"]["commit_events"] == 1
    assert transport.metrics()["counters"]["host_decode_steps"] == 0


def test_owner_lifecycle_parser_accepts_harmless_log_prefix() -> None:
    prefix, fields = _parse_fields(
        "[native] P5_OWNER_PHASE phase=compile_returned status=0"
    )
    assert prefix == "P5_OWNER_PHASE"
    assert fields == {"phase": "compile_returned", "status": "0"}


def test_owner_timeout_precedes_outer_service_timeout() -> None:
    assert OWNER_STARTUP_TIMEOUT_SECONDS < SERVICE_READINESS_TIMEOUT_SECONDS
    assert SERVICE_READINESS_TIMEOUT_SECONDS - OWNER_STARTUP_TIMEOUT_SECONDS >= 60


def test_owner_api_rejects_host_control_sampling_modes() -> None:
    settings = ServiceSettings("cruise-p5", ("owner",))
    valid = {
        "model": "cruise-p5",
        "prompt": list(range(1000, 1128)),
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
    }
    prompt, max_tokens, stream, ignore_eos, token_ids = _validate_request(
        valid, settings
    )
    assert (len(prompt), max_tokens, stream, ignore_eos, token_ids) == (
        128,
        256,
        True,
        True,
        True,
    )
    with pytest.raises(Exception):
        _validate_request({**valid, "temperature": 0.8}, settings)


class _FakeDecoder:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.text = ""

    def push(self, token: int) -> str:
        delta = f"<{token}>"
        self.tokens.append(token)
        self.text += delta
        return delta


class _FakeCodec:
    def __init__(self) -> None:
        self.prompts: list[tuple[int, ...]] = []

    def incremental(self, prompt: tuple[int, ...]) -> _FakeDecoder:
        self.prompts.append(prompt)
        return _FakeDecoder()


class _FakeTransport:
    instances: list["_FakeTransport"] = []

    def __init__(
        self, command: tuple[str, ...], *, startup_timeout: float = 1140.0
    ) -> None:
        self.command = command
        self.startup_timeout = startup_timeout
        self.closed = False
        self.admitted = 0
        self.instances.append(self)

    async def start(self) -> None:
        return None

    async def admit_many(
        self,
        requests: list[OwnerRequest],
        *,
        cohort_id: int,
        ignore_eos: bool,
        eos_token: int,
    ) -> None:
        del cohort_id, ignore_eos, eos_token
        self.admitted += len(requests)
        for request in requests:
            for index, token in enumerate((41, 42), start=1):
                request.events.put_nowait(
                    OwnerEvent(
                        kind="commit",
                        request=request.request,
                        generation=request.generation,
                        row=0,
                        commit_seq=index,
                        token=token,
                        position=len(request.prompt_tokens) + index,
                        page=0,
                        status=0,
                        aicore_calls=index,
                        total_commits=index,
                        total_retired=0,
                        checksum=index,
                        finish_reason=0,
                    )
                )
            request.retired = True
            request.events.put_nowait(
                OwnerEvent(
                    kind="retire_complete",
                    request=request.request,
                    generation=request.generation,
                    row=0,
                    commit_seq=2,
                    token=42,
                    position=len(request.prompt_tokens) + 2,
                    page=0,
                    status=0,
                    aicore_calls=2,
                    total_commits=2,
                    total_retired=1,
                    checksum=2,
                    finish_reason=2,
                )
            )

    async def cancel(self, request: OwnerRequest) -> None:
        request.retired = True

    async def close(self) -> None:
        self.closed = True

    def metrics(self) -> dict[str, object]:
        return {
            "route": "persistent_device_model_owner",
            "forbidden_runtime_modules": [],
            "counters": {
                "admission_events": self.admitted,
                "host_decode_steps": 0,
            },
        }


def test_owner_service_streams_incremental_tokens_through_fake_transport() -> None:
    _FakeTransport.instances.clear()
    codec = _FakeCodec()
    app = create_app(
        ServiceSettings("cruise-p5", ("fake-owner",), batch_wait_ms=0),
        codec,  # type: ignore[arg-type]
        _FakeTransport,  # type: ignore[arg-type]
    )
    body = {
        "model": "cruise-p5",
        "prompt": [1000, 1001],
        "max_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    with TestClient(app) as client:
        response = client.post("/v1/completions", json=body)
        assert response.status_code == 200
        lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        payloads = [json.loads(line[6:]) for line in lines[:-1]]
        choices = [choice for payload in payloads for choice in payload["choices"]]
        assert [choice["token_ids"] for choice in choices] == [[41], [42], []]
        assert "".join(choice["text"] for choice in choices) == "<41><42>"
        assert payloads[-1]["usage"] == {
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "total_tokens": 4,
        }
    assert codec.prompts == [(1000, 1001)]
    assert _FakeTransport.instances[0].closed is True


def test_owner_service_nonstreaming_uses_incremental_decoder() -> None:
    codec = _FakeCodec()
    app = create_app(
        ServiceSettings("cruise-p5", ("fake-owner",), batch_wait_ms=0),
        codec,  # type: ignore[arg-type]
        _FakeTransport,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "cruise-p5",
                "prompt": [1000],
                "max_tokens": 2,
                "temperature": 0.0,
                "stream": False,
                "return_token_ids": True,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["text"] == "<41><42>"
    assert payload["choices"][0]["token_ids"] == [41, 42]
    assert payload["choices"][0]["finish_reason"] == "length"


def test_model_revision_requires_independent_identity_marker(tmp_path: Path) -> None:
    model = tmp_path / "Qwen2.5-7B-Instruct"
    model.mkdir()
    assert _verified_revision(model, "revision-1") is False
    (model / ".cruise-model-revision").write_text("revision-1\n", encoding="utf-8")
    assert _verified_revision(model, "revision-1") is True


def test_process_tree_cpu_meter_reports_its_measurement_window() -> None:
    meter = ProcessTreeCpuMeter(os.getpid(), interval=0.002)
    meter.start()
    deadline = time.perf_counter() + 0.03
    value = 0
    while time.perf_counter() < deadline:
        value += 1
    result = meter.stop()
    assert value > 0
    assert result["observed_process_identities"] >= 1
    assert result["wall_duration_ns"] > 0
    assert result["sample_count"] >= 2


def test_failed_start_diagnostic_tail_is_strictly_bounded(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    logs = runtime / "cann-logs"
    logs.mkdir(parents=True)
    for index in range(3):
        (logs / f"device-{index}.log").write_bytes(bytes([65 + index]) * 64)
    output = tmp_path / "diagnostic-owner-1.log"
    result = _write_diagnostic_tail(
        runtime, output, max_files=2, max_bytes_per_file=16
    )
    assert result is not None
    assert result["source_files"] == 2
    assert result["retained_source_bytes"] == 32
    assert output.stat().st_size < 512


def _synthetic_start(label: str) -> dict[str, object]:
    route = label.split("-", 1)[0]
    owner = route == "owner"
    records = [
        {
            "request_index": index,
            "tokens": [42, 43],
            "text": "answer",
            "finish_reason": "length",
            "done": True,
            "tpot_ms": 8.0 if owner else 10.0,
            "ttft_ms": 5.0,
            "usage": {
                "prompt_tokens": 128,
                "completion_tokens": 256,
                "total_tokens": 384,
            },
            "stream_chunk_sizes": [1] * 256,
        }
        for index in range(32)
    ]
    cpu_seconds = 0.4 if owner else 1.0
    result: dict[str, object] = {
        "route": route,
        "run_label": label,
        "start_uuid": label + "-uuid",
        "model_manifest_sha256": (
            "651d64436c415afd6faf3d82f14086c2df54d99c02f66f81a15b83a2d17de5f9"
        ),
        "pass": True,
        "primary": records,
        "metrics": {
            "duration_ms": 640.0 if owner else 800.0,
            "output_tokens_per_second": 100.0 if owner else 80.0,
            "host_cpu_ms_per_output_token": cpu_seconds * 1000 / 64,
        },
        "process_tree_cpu": {
            "method": "sampled-/proc whole-process-tree utime+stime",
            "observed_process_identities": 2,
            "cpu_seconds": cpu_seconds,
            "wall_duration_ns": 650_000_000 if owner else 810_000_000,
            "load_duration_ns": 640_000_000 if owner else 800_000_000,
        },
    }
    if owner:
        result["owner_metrics"] = {
            "route": "persistent_device_model_owner",
            "host_visible_decode_epoch": False,
            "host_token_step_api": False,
            "forbidden_runtime_modules": [],
        }
        result["owner_counter_delta"] = {
            "admission_events": 32,
            "aicore_calls": 3064,
            "commit_events": 8192,
            "host_decode_steps": 0,
        }
    return result


def test_p5_verifier_accepts_complete_qualified_matrix(tmp_path: Path) -> None:
    paths = []
    for label in START_ORDER:
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(_synthetic_start(label)), encoding="utf-8")
        paths.append(path)
    result = verify(paths, P5 / "workload.json")
    assert result["execution_pass"] is True
    assert result["qualification_pass"] is True
    assert result["decision"] == "P5-Persistent-Decode-Qualified"


def test_p5_first_pair_gate_accepts_exact_formal_smoke(tmp_path: Path) -> None:
    graph = tmp_path / "graph-1.json"
    owner = tmp_path / "owner-1.json"
    graph.write_text(json.dumps(_synthetic_start("graph-1")), encoding="utf-8")
    owner.write_text(json.dumps(_synthetic_start("owner-1")), encoding="utf-8")
    result = verify_pair(graph, owner)
    assert result["pass"] is True
    assert result["decision"] == "continue-six-start-matrix"


def test_p5_first_pair_gate_stops_on_usage_mismatch(tmp_path: Path) -> None:
    graph_result = _synthetic_start("graph-1")
    owner_result = _synthetic_start("owner-1")
    owner_result["primary"][0]["usage"]["completion_tokens"] = 255  # type: ignore[index]
    graph = tmp_path / "graph-1.json"
    owner = tmp_path / "owner-1.json"
    graph.write_text(json.dumps(graph_result), encoding="utf-8")
    owner.write_text(json.dumps(owner_result), encoding="utf-8")
    result = verify_pair(graph, owner)
    assert result["pass"] is False
    assert result["checks"]["semantics_exact"] is False
