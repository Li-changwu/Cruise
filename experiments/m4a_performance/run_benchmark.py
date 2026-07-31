#!/usr/bin/env python3
"""Run and compare the three-route M4a API performance preflight."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import urlopen

from experiments.m1_batched_prefill.run_differential import (
    SCHEDULER_QUALNAME,
    WORKER_QUALNAME,
    configured_kv_cache_bytes,
)


MODES = ("eager", "graph", "cruise")
BLOCKED_ORDER = (
    "eager-1",
    "graph-1",
    "cruise-1",
    "cruise-2",
    "graph-2",
    "eager-2",
    "graph-3",
    "cruise-3",
    "eager-3",
)
REQUIRED_SCENARIOS = {
    "short-stream-c1",
    "decode-stream-c1",
    "decode-stream-c4",
    "decode-nonstream-c1",
    "decode-nonstream-c4",
    "decode-bursty-c4",
    "decode-overload-c8",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    stream: bool
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    request_count: int
    concurrency: int
    arrival: str
    burst_gap_ms: int = 0

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stream": self.stream,
            "prompt_token_ids": list(self.prompt_token_ids),
            "max_tokens": self.max_tokens,
            "request_count": self.request_count,
            "concurrency": self.concurrency,
            "arrival": self.arrival,
            "burst_gap_ms": self.burst_gap_ms,
        }


@dataclass(frozen=True)
class PerformanceManifest:
    served_model_name: str
    tokenizer: Path
    warmups: tuple[Scenario, ...]
    scenarios: tuple[Scenario, ...]
    primary_scenario: str
    thresholds: dict[str, float]

    def expected_device_request_tokens(self) -> int:
        return sum(
            (scenario.max_tokens - 1) * scenario.request_count
            for scenario in (*self.warmups, *self.scenarios)
        )


def _parse_scenario(raw: Any, names: set[str]) -> Scenario:
    if not isinstance(raw, dict):
        raise ValueError("each M4a scenario must be an object")
    name = raw.get("name")
    stream = raw.get("stream")
    prompt = raw.get("prompt_token_ids")
    max_tokens = raw.get("max_tokens")
    request_count = raw.get("request_count")
    concurrency = raw.get("concurrency")
    arrival = raw.get("arrival")
    burst_gap_ms = raw.get("burst_gap_ms", 0)
    if not isinstance(name, str) or not name or name in names:
        raise ValueError("M4a scenario names must be non-empty and unique")
    names.add(name)
    if not isinstance(stream, bool):
        raise ValueError(f"{name}: stream must be boolean")
    if (
        not isinstance(prompt, list)
        or not 2 <= len(prompt) <= 5
        or not all(isinstance(token, int) and token >= 0 for token in prompt)
    ):
        raise ValueError(f"{name}: prompt_token_ids are invalid")
    if not isinstance(max_tokens, int) or not 2 <= max_tokens <= 7:
        raise ValueError(f"{name}: max_tokens must be in [2, 7]")
    if len(prompt) + max_tokens - 1 > 8:
        raise ValueError(f"{name}: request exceeds resident logical capacity")
    if not isinstance(request_count, int) or request_count < 8:
        raise ValueError(f"{name}: request_count must be at least eight")
    if concurrency not in (1, 2, 4, 8) or request_count % concurrency != 0:
        raise ValueError(f"{name}: invalid concurrency or request count")
    if arrival not in ("closed_loop", "bursty"):
        raise ValueError(f"{name}: arrival must be closed_loop or bursty")
    if not isinstance(burst_gap_ms, int) or burst_gap_ms < 0:
        raise ValueError(f"{name}: burst_gap_ms must be a non-negative integer")
    if arrival == "bursty" and burst_gap_ms == 0:
        raise ValueError(f"{name}: bursty arrival requires a positive gap")
    if arrival == "closed_loop" and burst_gap_ms != 0:
        raise ValueError(f"{name}: closed_loop arrival cannot specify a gap")
    return Scenario(
        name=name,
        stream=stream,
        prompt_token_ids=tuple(prompt),
        max_tokens=max_tokens,
        request_count=request_count,
        concurrency=concurrency,
        arrival=arrival,
        burst_gap_ms=burst_gap_ms,
    )


def load_manifest(path: Path) -> PerformanceManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported M4a workload schema")
    model_name = payload.get("served_model_name")
    tokenizer = payload.get("tokenizer")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("served_model_name must be non-empty")
    if not isinstance(tokenizer, str) or not tokenizer:
        raise ValueError("tokenizer must be a path")
    names: set[str] = set()
    raw_warmups = payload.get("warmups")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_warmups, list) or len(raw_warmups) != 2:
        raise ValueError("M4a requires streaming and non-streaming warmups")
    if not isinstance(raw_scenarios, list):
        raise ValueError("M4a scenarios must be a list")
    warmups = tuple(_parse_scenario(raw, names) for raw in raw_warmups)
    scenarios = tuple(_parse_scenario(raw, names) for raw in raw_scenarios)
    if {scenario.name for scenario in scenarios} != REQUIRED_SCENARIOS:
        raise ValueError("the M4a scenario set is incomplete")
    primary = payload.get("primary_scenario")
    if primary != "decode-stream-c4":
        raise ValueError("the M4a primary scenario must be decode-stream-c4")
    thresholds = payload.get("thresholds")
    expected_thresholds = {
        "median_tpot_improvement_percent": 15.0,
        "p95_tpot_improvement_percent": 15.0,
        "host_cpu_per_token_reduction_percent": 30.0,
    }
    if thresholds != expected_thresholds:
        raise ValueError("M4a thresholds must match the frozen protocol")
    return PerformanceManifest(
        served_model_name=model_name,
        tokenizer=Path(tokenizer),
        warmups=warmups,
        scenarios=scenarios,
        primary_scenario=primary,
        thresholds=expected_thresholds,
    )


def with_tokenizer_override(manifest: PerformanceManifest) -> PerformanceManifest:
    override = os.getenv("CRUISE_API_TOKENIZER")
    if override is None:
        return manifest
    return PerformanceManifest(
        served_model_name=manifest.served_model_name,
        tokenizer=Path(override),
        warmups=manifest.warmups,
        scenarios=manifest.scenarios,
        primary_scenario=manifest.primary_scenario,
        thresholds=manifest.thresholds,
    )


def percentile(values: Iterable[float], percent: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    sample = [float(value) for value in values]
    if not sample:
        return {"count": 0}
    return {
        "count": len(sample),
        "min": min(sample),
        "mean": sum(sample) / len(sample),
        "p50": percentile(sample, 50.0),
        "p95": percentile(sample, 95.0),
        "p99": percentile(sample, 99.0),
        "max": max(sample),
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(base_url: str, process: subprocess.Popen[Any], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API server exited during startup: {process.returncode}")
        try:
            with urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
                last_error = f"health returned {response.status}"
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"API server readiness timed out: {last_error}")


def server_command(
    *, mode: str, model: Path, manifest: PerformanceManifest, port: int
) -> list[str]:
    if mode not in MODES:
        raise ValueError(f"unknown M4a route: {mode}")
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        str(model),
        "--tokenizer",
        str(manifest.tokenizer),
        "--served-model-name",
        manifest.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--generation-config",
        "vllm",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "128",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "128",
        "--gpu-memory-utilization",
        "0.35",
        "--kv-cache-memory-bytes",
        str(configured_kv_cache_bytes()),
        "--worker-cls",
        WORKER_QUALNAME,
        "--no-async-scheduling",
    ]
    if mode == "eager":
        command.append("--enforce-eager")
    elif mode == "cruise":
        command.extend(("--scheduler-cls", SCHEDULER_QUALNAME))
    return command


def _bounded_logger_command(output: Path) -> tuple[list[str], Path]:
    logger = Path(__file__).resolve().parents[2] / "storage_guard" / "bounded_log.py"
    metadata = output.with_suffix(".meta.json")
    return (
        [
            sys.executable,
            str(logger),
            "--output",
            str(output),
            "--metadata",
            str(metadata),
            "--head-bytes",
            str(1024 * 1024),
            "--tail-bytes",
            str(1024 * 1024),
        ],
        metadata,
    )


def _stop_server(process: subprocess.Popen[Any]) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    assert process.returncode is not None
    return int(process.returncode)


def _process_tree_snapshot(root_pid: int) -> dict[str, float | int]:
    proc = Path("/proc")
    if not proc.is_dir():
        return {"cpu_seconds": 0.0, "rss_bytes": 0, "process_count": 0}
    ticks = int(os.sysconf("SC_CLK_TCK"))
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    table: dict[int, tuple[int, float, int]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            remainder = stat[stat.rfind(")") + 2 :].split()
            pid = int(entry.name)
            ppid = int(remainder[1])
            cpu_seconds = (int(remainder[11]) + int(remainder[12])) / ticks
            resident_pages = int(
                (entry / "statm").read_text(encoding="utf-8").split()[1]
            )
            table[pid] = (ppid, cpu_seconds, resident_pages * page_size)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _) in table.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    records = [table[pid] for pid in descendants if pid in table]
    return {
        "cpu_seconds": sum(record[1] for record in records),
        "rss_bytes": sum(record[2] for record in records),
        "process_count": len(records),
    }


def _npu_usage_snapshot() -> str:
    physical_npu = os.getenv(
        "CRUISE_PHYSICAL_NPU", os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    )
    try:
        completed = subprocess.run(
            ["npu-smi", "info", "-t", "usages", "-i", physical_npu],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    rendered = (completed.stdout + completed.stderr).strip()
    return rendered[:8192]


def _completion_request_body(
    model_name: str, scenario: Scenario
) -> dict[str, Any]:
    return {
        "model": model_name,
        "prompt": list(scenario.prompt_token_ids),
        "max_tokens": scenario.max_tokens,
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


async def _request(
    client: Any,
    base_url: str,
    model_name: str,
    scenario: Scenario,
    request_index: int,
) -> dict[str, Any]:
    body = _completion_request_body(model_name, scenario)
    started = time.perf_counter_ns()
    token_times_ns: list[int] = []
    tokens: list[int] = []
    finish_reason = None
    stop_reason = None
    usage = None
    done = not scenario.stream
    try:
        if scenario.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            async with client.stream(
                "POST", f"{base_url}/v1/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        done = True
                        break
                    payload = json.loads(data)
                    if payload.get("usage") is not None:
                        usage = payload["usage"]
                    for choice in payload.get("choices", []):
                        if choice.get("index") != 0:
                            raise ValueError("single-prompt M4a response used a nonzero index")
                        new_tokens = list(choice.get("token_ids") or [])
                        timestamp = time.perf_counter_ns()
                        tokens.extend(new_tokens)
                        token_times_ns.extend(timestamp for _ in new_tokens)
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]
                            stop_reason = choice.get("stop_reason")
        else:
            response = await client.post(f"{base_url}/v1/completions", json=body)
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices", [])
            if len(choices) != 1 or choices[0].get("index") != 0:
                raise ValueError("M4a non-stream response must contain choice zero")
            tokens = list(choices[0].get("token_ids") or [])
            finish_reason = choices[0].get("finish_reason")
            stop_reason = choices[0].get("stop_reason")
            usage = payload.get("usage")
    except Exception as exc:
        ended = time.perf_counter_ns()
        return {
            "request_index": request_index,
            "tokens": tokens,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "done": done,
            "latency_ms": (ended - started) / 1_000_000,
            "error": f"{type(exc).__name__}: {exc}",
            "pass": False,
        }

    ended = time.perf_counter_ns()
    ttft_ms = (
        (token_times_ns[0] - started) / 1_000_000 if token_times_ns else None
    )
    inter_token_ms = [
        (right - left) / 1_000_000
        for left, right in zip(token_times_ns, token_times_ns[1:])
    ]
    tpot_ms = (
        (token_times_ns[-1] - token_times_ns[0])
        / (len(token_times_ns) - 1)
        / 1_000_000
        if len(token_times_ns) > 1
        else None
    )
    checks = {
        "exact_output_length": len(tokens) == scenario.max_tokens,
        "length_finish": finish_reason == "length",
        "done_boundary": done,
    }
    return {
        "request_index": request_index,
        "tokens": tokens,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "usage": usage,
        "done": done,
        "latency_ms": (ended - started) / 1_000_000,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "inter_token_ms": inter_token_ms,
        "checks": checks,
        "pass": all(checks.values()),
    }


async def _execute_scenario(
    client: Any,
    base_url: str,
    model_name: str,
    scenario: Scenario,
) -> list[dict[str, Any]]:
    if scenario.arrival == "bursty":
        records: list[dict[str, Any]] = []
        for offset in range(0, scenario.request_count, scenario.concurrency):
            records.extend(
                await asyncio.gather(
                    *(
                        _request(
                            client,
                            base_url,
                            model_name,
                            scenario,
                            request_index,
                        )
                        for request_index in range(
                            offset, offset + scenario.concurrency
                        )
                    )
                )
            )
            if offset + scenario.concurrency < scenario.request_count:
                await asyncio.sleep(scenario.burst_gap_ms / 1000.0)
        return sorted(records, key=lambda record: record["request_index"])

    semaphore = asyncio.Semaphore(scenario.concurrency)

    async def limited(request_index: int) -> dict[str, Any]:
        async with semaphore:
            return await _request(
                client, base_url, model_name, scenario, request_index
            )

    records = await asyncio.gather(
        *(limited(index) for index in range(scenario.request_count))
    )
    return sorted(records, key=lambda record: record["request_index"])


def _scenario_metrics(
    records: list[dict[str, Any]],
    *,
    duration_ms: float,
    cpu_seconds: float,
) -> dict[str, Any]:
    output_tokens = sum(len(record.get("tokens", [])) for record in records)
    latency = [record["latency_ms"] for record in records]
    ttft = [record["ttft_ms"] for record in records if record.get("ttft_ms") is not None]
    tpot = [record["tpot_ms"] for record in records if record.get("tpot_ms") is not None]
    inter_token = [
        value for record in records for value in record.get("inter_token_ms", [])
    ]
    normalized = [
        record["latency_ms"] / len(record["tokens"])
        for record in records
        if record.get("tokens")
    ]
    duration_seconds = duration_ms / 1000.0
    return {
        "request_latency_ms": summarize(latency),
        "ttft_ms": summarize(ttft),
        "tpot_ms": summarize(tpot),
        "inter_token_ms": summarize(inter_token),
        "normalized_request_ms_per_output_token": summarize(normalized),
        "duration_ms": duration_ms,
        "request_count": len(records),
        "output_tokens": output_tokens,
        "requests_per_second": len(records) / duration_seconds,
        "output_tokens_per_second": output_tokens / duration_seconds,
        "host_cpu_seconds": cpu_seconds,
        "host_cpu_ms_per_output_token": (
            cpu_seconds * 1000.0 / output_tokens if output_tokens else None
        ),
    }


async def _run_load(
    base_url: str,
    manifest: PerformanceManifest,
    process: subprocess.Popen[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import httpx

    timeout = httpx.Timeout(600.0, connect=30.0)
    warmup_results: list[dict[str, Any]] = []
    scenario_results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for scenario in manifest.warmups:
            records = await _execute_scenario(
                client, base_url, manifest.served_model_name, scenario
            )
            warmup_results.append(
                {
                    "scenario": scenario.as_record(),
                    "request_count": len(records),
                    "all_requests_passed": all(record["pass"] for record in records),
                }
            )
        for scenario in manifest.scenarios:
            npu_before = _npu_usage_snapshot()
            process_before = _process_tree_snapshot(process.pid)
            started = time.perf_counter_ns()
            records = await _execute_scenario(
                client, base_url, manifest.served_model_name, scenario
            )
            ended = time.perf_counter_ns()
            process_after = _process_tree_snapshot(process.pid)
            npu_after = _npu_usage_snapshot()
            cpu_seconds = max(
                0.0,
                float(process_after["cpu_seconds"])
                - float(process_before["cpu_seconds"]),
            )
            duration_ms = (ended - started) / 1_000_000
            scenario_results.append(
                {
                    "scenario": scenario.as_record(),
                    "records": records,
                    "metrics": _scenario_metrics(
                        records, duration_ms=duration_ms, cpu_seconds=cpu_seconds
                    ),
                    "process_tree_before": process_before,
                    "process_tree_after": process_after,
                    "npu_usage_before": npu_before,
                    "npu_usage_after": npu_after,
                    "pass": len(records) == scenario.request_count
                    and all(record["pass"] for record in records),
                }
            )
    return warmup_results, scenario_results


def _mode_identity(mode: str, server_log: str, route_metrics: Any) -> dict[str, bool]:
    if mode == "eager":
        return {
            "enforce_eager_true": "enforce_eager=True" in server_log,
            "no_cruise_scheduler": "--scheduler-cls" not in server_log,
        }
    graph_checks = {
        "enforce_eager_false": "enforce_eager=False" in server_log,
        "piecewise_aclgraph_enabled": "PIECEWISE compilation enabled on NPU"
        in server_log,
        "aclgraph_replayed": "Replaying aclgraph" in server_log,
    }
    if mode == "graph":
        return graph_checks
    counters = route_metrics.get("counters", {}) if isinstance(route_metrics, dict) else {}
    return {
        **graph_checks,
        "device_epochs_observed": counters.get("device_epochs", 0) > 0,
        "one_feed_per_device_epoch": counters.get("feed_calls")
        == counters.get("device_epochs"),
        "one_fetch_per_device_epoch": counters.get("fetch_calls")
        == counters.get("device_epochs"),
    }


def run_service(
    *,
    mode: str,
    run_label: str,
    model: Path,
    manifest: PerformanceManifest,
    output: Path,
    runtime_dir: Path,
) -> dict[str, Any]:
    if not manifest.tokenizer.is_dir():
        raise FileNotFoundError(f"tokenizer directory not found: {manifest.tokenizer}")
    runtime_dir.mkdir(parents=True, exist_ok=False)
    for child in ("cache", "cann-logs", "tmp"):
        (runtime_dir / child).mkdir()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server_log = output.with_name(f"api-server-{run_label}.log")
    logger_command, logger_metadata = _bounded_logger_command(server_log)
    route_metrics_path = runtime_dir / "resident-route-metrics.json"
    command = server_command(mode=mode, model=model, manifest=manifest, port=port)
    server_env = os.environ.copy()
    server_env.update(
        {
            "ASCEND_CACHE_PATH": str(runtime_dir / "cache"),
            "ASCEND_PROCESS_LOG_PATH": str(runtime_dir / "cann-logs"),
            "TORCHINDUCTOR_CACHE_DIR": str(runtime_dir / "cache" / "torchinductor"),
            "TRITON_CACHE_DIR": str(runtime_dir / "cache" / "triton"),
            "XDG_CACHE_HOME": str(runtime_dir / "cache" / "xdg"),
            "TMPDIR": str(runtime_dir / "tmp"),
            "VLLM_ASCEND_RESIDENT_EPOCH_SOCKET": str(runtime_dir / "control.sock"),
        }
    )
    if mode == "cruise":
        server_env[
            "VLLM_ASCEND_RESIDENT_EPOCH_BENCHMARK_METRICS_PATH"
        ] = str(route_metrics_path)
    else:
        server_env.pop("VLLM_ASCEND_RESIDENT_EPOCH_BENCHMARK_METRICS_PATH", None)

    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": "M4a three-route API performance preflight",
        "mode": mode,
        "run_label": run_label,
        "model": str(model),
        "tokenizer": str(manifest.tokenizer),
        "command": command,
        "runtime_dir": str(runtime_dir),
        "server_log": str(server_log),
        "server_log_metadata": str(logger_metadata),
        "warmups": [],
        "scenarios": [],
        "pass": False,
    }
    process: subprocess.Popen[Any] | None = None
    logger_process: subprocess.Popen[Any] | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        logger_process = subprocess.Popen(logger_command, stdin=subprocess.PIPE)
        assert logger_process.stdin is not None
        initialized_at = time.perf_counter_ns()
        process = subprocess.Popen(
            command,
            stdout=logger_process.stdin,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=server_env,
        )
        logger_process.stdin.close()
        _wait_ready(base_url, process, 1200)
        result["initialization_ms"] = (
            time.perf_counter_ns() - initialized_at
        ) / 1_000_000
        result["warmups"], result["scenarios"] = asyncio.run(
            _run_load(base_url, manifest, process)
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if process is not None:
            try:
                result["server_returncode"] = _stop_server(process)
            except Exception as exc:
                result["shutdown_error"] = f"{type(exc).__name__}: {exc}"
        if logger_process is not None:
            try:
                result["logger_returncode"] = logger_process.wait(timeout=60)
            except Exception as exc:
                logger_process.kill()
                logger_process.wait(timeout=10)
                result["logger_error"] = f"{type(exc).__name__}: {exc}"

    route_metrics = None
    if route_metrics_path.is_file():
        route_metrics = json.loads(route_metrics_path.read_text(encoding="utf-8"))
    result["resident_route_metrics"] = route_metrics
    log_text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.is_file() else ""
    result["mode_identity"] = _mode_identity(mode, log_text, route_metrics)
    result["checks"] = {
        "all_warmups_passed": len(result["warmups"]) == len(manifest.warmups)
        and all(warmup["all_requests_passed"] for warmup in result["warmups"]),
        "all_scenarios_passed": len(result["scenarios"]) == len(manifest.scenarios)
        and all(scenario["pass"] for scenario in result["scenarios"]),
        "mode_identity_proven": bool(result["mode_identity"])
        and all(result["mode_identity"].values()),
        "clean_server_exit": result.get("server_returncode") == 0,
        "clean_logger_exit": result.get("logger_returncode") == 0,
    }
    result["pass"] = all(result["checks"].values())
    return result


def _semantics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        item["scenario"]["name"]: [
            {
                "request_index": record["request_index"],
                "tokens": record.get("tokens"),
                "finish_reason": record.get("finish_reason"),
                "stop_reason": record.get("stop_reason"),
                "done": record.get("done"),
            }
            for record in item.get("records", [])
        ]
        for item in result.get("scenarios", [])
    }


def _aggregate_scenario(results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    scenarios = [
        next(item for item in result["scenarios"] if item["scenario"]["name"] == name)
        for result in results
    ]
    records = [record for scenario in scenarios for record in scenario["records"]]
    duration_ms = sum(float(scenario["metrics"]["duration_ms"]) for scenario in scenarios)
    cpu_seconds = sum(
        float(scenario["metrics"]["host_cpu_seconds"]) for scenario in scenarios
    )
    aggregate = _scenario_metrics(
        records, duration_ms=duration_ms, cpu_seconds=cpu_seconds
    )
    aggregate["independent_starts"] = len(results)
    aggregate["per_start_output_tokens_per_second"] = summarize(
        scenario["metrics"]["output_tokens_per_second"] for scenario in scenarios
    )
    aggregate["per_start_host_cpu_ms_per_output_token"] = summarize(
        scenario["metrics"]["host_cpu_ms_per_output_token"] for scenario in scenarios
    )
    return aggregate


def _improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline metric must be positive")
    return (baseline - candidate) / baseline * 100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_results(
    paths: list[Path], manifest: PerformanceManifest
) -> dict[str, Any]:
    loaded = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    labels = [result.get("run_label") for _, result in loaded]
    grouped = {
        mode: [result for _, result in loaded if result.get("mode") == mode]
        for mode in MODES
    }
    aggregates = {
        mode: {
            scenario.name: _aggregate_scenario(grouped[mode], scenario.name)
            for scenario in manifest.scenarios
        }
        for mode in MODES
        if len(grouped[mode]) == 3
    }

    canonical = _semantics(loaded[0][1]) if loaded else {}
    semantic_mismatches = {
        result.get("run_label", str(path)): _semantics(result)
        for path, result in loaded
        if _semantics(result) != canonical
    }
    primary = manifest.primary_scenario
    baseline_mode = None
    threshold_values: dict[str, Any] = {}
    if all(mode in aggregates for mode in MODES):
        baseline_mode = min(
            ("graph", "eager"),
            key=lambda mode: (
                aggregates[mode][primary]["tpot_ms"]["p50"],
                0 if mode == "graph" else 1,
            ),
        )
        baseline = aggregates[baseline_mode][primary]
        cruise = aggregates["cruise"][primary]
        threshold_values = {
            "strongest_baseline": baseline_mode,
            "median_tpot_improvement_percent": _improvement(
                float(baseline["tpot_ms"]["p50"]),
                float(cruise["tpot_ms"]["p50"]),
            ),
            "p95_tpot_improvement_percent": _improvement(
                float(baseline["tpot_ms"]["p95"]),
                float(cruise["tpot_ms"]["p95"]),
            ),
            "host_cpu_per_token_reduction_percent": _improvement(
                float(baseline["host_cpu_ms_per_output_token"]),
                float(cruise["host_cpu_ms_per_output_token"]),
            ),
        }

    threshold_checks = {
        key: threshold_values.get(key, float("-inf")) >= expected
        for key, expected in manifest.thresholds.items()
    }
    expected_tokens = manifest.expected_device_request_tokens()
    cruise_route_checks: dict[str, Any] = {}
    if len(grouped["cruise"]) == 3:
        route_records = [result.get("resident_route_metrics") for result in grouped["cruise"]]
        counters = [
            record.get("counters", {}) if isinstance(record, dict) else {}
            for record in route_records
        ]
        cruise_route_checks = {
            "expected_device_request_tokens_per_start": expected_tokens,
            "observed_device_request_tokens": [
                counter.get("device_request_tokens", 0) for counter in counters
            ],
            "route_hit_rates": [
                counter.get("device_request_tokens", 0) / expected_tokens
                for counter in counters
            ],
            "all_eligible_decode_tokens_used_device": all(
                counter.get("device_request_tokens") == expected_tokens
                for counter in counters
            ),
        }

    execution_checks = {
        "blocked_order_exact": labels == list(BLOCKED_ORDER),
        "three_independent_starts_per_mode": all(
            len(grouped[mode]) == 3 for mode in MODES
        ),
        "all_runs_passed": len(loaded) == 9
        and all(result.get("pass") is True for _, result in loaded),
        "exact_api_semantics": not semantic_mismatches,
        "cruise_route_coverage": cruise_route_checks.get(
            "all_eligible_decode_tokens_used_device"
        )
        is True,
    }
    execution_pass = all(execution_checks.values())
    qualification_pass = execution_pass and all(threshold_checks.values())
    return {
        "schema_version": 1,
        "gate": "M4a performance preflight comparison",
        "formal_milestones_closed": [],
        "input_sha256": {str(path): _sha256(path) for path, _ in loaded},
        "run_labels": labels,
        "aggregates": aggregates,
        "primary_scenario": primary,
        "thresholds": manifest.thresholds,
        "threshold_values": threshold_values,
        "threshold_checks": threshold_checks,
        "resident_route_coverage": cruise_route_checks,
        "semantic_mismatches": semantic_mismatches,
        "execution_checks": execution_checks,
        "execution_pass": execution_pass,
        "qualification_pass": qualification_pass,
        "decision": (
            "return-to-m2-m3-before-formal-m4"
            if qualification_pass
            else "performance-attribution-required"
        ),
        "pass": execution_pass,
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(*MODES, "compare"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--result", action="append", type=Path, default=[])
    args = parser.parse_args()

    manifest = with_tokenizer_override(
        load_manifest(args.workload.resolve(strict=True))
    )
    if args.mode == "compare":
        if len(args.result) != 9:
            parser.error("compare mode requires nine ordered --result paths")
        result = compare_results(
            [path.resolve(strict=True) for path in args.result], manifest
        )
    else:
        if args.model is None or args.run_label is None or args.runtime_dir is None:
            parser.error("service mode requires --model, --run-label and --runtime-dir")
        if not args.run_label.startswith(f"{args.mode}-"):
            parser.error("run label must start with its mode")
        result = run_service(
            mode=args.mode,
            run_label=args.run_label,
            model=args.model.resolve(strict=True),
            manifest=manifest,
            output=args.output,
            runtime_dir=args.runtime_dir,
        )
    write_result(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
