#!/usr/bin/env python3
"""Run one cold P5 Graph or Persistent Owner service start."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROUTES = ("graph", "owner")
START_ORDER = (
    "graph-1",
    "owner-1",
    "owner-2",
    "graph-2",
    "graph-3",
    "owner-3",
)
SERVICE_READINESS_TIMEOUT_SECONDS = 1200
OWNER_STARTUP_TIMEOUT_SECONDS = 1140
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_MANIFEST_SHA256 = (
    "651d64436c415afd6faf3d82f14086c2df54d99c02f66f81a15b83a2d17de5f9"
)
SAMPLE_INTERVAL_SECONDS = 0.02
GRAPH_PLUGIN_ALLOWLIST = (
    "ascend",
    "ascend_kv_connector",
    "ascend_model",
    "ascend_model_loader",
    "ascend_service_profiling",
)
GRAPH_ASCEND_ADDITIONAL_CONFIG = json.dumps(
    {"ascend_compilation_config": {"fuse_norm_quant": False}},
    separators=(",", ":"),
)
GRAPH_RMSNORM_BACKEND = "torch_npu.npu_add_rms_norm"


@dataclass(frozen=True)
class Workload:
    model: str
    revision: str
    model_manifest_sha256: str
    served_model_name: str
    warmup: dict[str, Any]
    primary: dict[str, Any]
    start_order: tuple[str, ...]
    thresholds: dict[str, float]


def load_workload(path: Path) -> Workload:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_thresholds = {
        "host_cpu_per_token_reduction_percent": 50.0,
        "tpot_p50_improvement_percent": 15.0,
        "tpot_p95_improvement_percent": 15.0,
        "output_throughput_improvement_percent": 15.0,
    }
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported P5 workload schema")
    if payload.get("revision") != REVISION:
        raise ValueError("P5 must use the pinned Qwen2.5-7B revision")
    if payload.get("model_manifest_sha256") != MODEL_MANIFEST_SHA256:
        raise ValueError("P5 must use the pinned canonical model manifest")
    if tuple(payload.get("start_order", ())) != START_ORDER:
        raise ValueError("P5 start order differs from the frozen interleaving")
    if payload.get("thresholds") != expected_thresholds:
        raise ValueError("P5 thresholds differ from ADR 0017")
    warmup = payload.get("warmup")
    primary = payload.get("primary")
    if not isinstance(warmup, dict) or not isinstance(primary, dict):
        raise ValueError("P5 workload must declare warmup and primary objects")
    expected_primary = {
        "output_tokens": 256,
        "request_count": 32,
        "concurrency": 4,
        "streaming": True,
        "ignore_eos": True,
    }
    for key, expected in expected_primary.items():
        if primary.get(key) != expected:
            raise ValueError(f"P5 primary {key} must be {expected!r}")
    prompt = primary.get("prompt_token_ids")
    if prompt != list(range(1000, 1128)):
        raise ValueError("P5 primary prompt must be the frozen 128-token vector")
    if (
        warmup.get("request_count") != 4
        or warmup.get("concurrency") != 4
        or warmup.get("streaming") is not True
        or warmup.get("ignore_eos") is not True
    ):
        raise ValueError("P5 warmup must be one streaming C4 cohort")
    return Workload(
        model=str(payload["model"]),
        revision=str(payload["revision"]),
        model_manifest_sha256=str(payload["model_manifest_sha256"]),
        served_model_name=str(payload["served_model_name"]),
        warmup=warmup,
        primary=primary,
        start_order=START_ORDER,
        thresholds=expected_thresholds,
    )


def percentile(values: Iterable[float], percent: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    sample = [float(value) for value in values]
    if not sample:
        return {"count": 0}
    return {
        "count": len(sample),
        "min": min(sample),
        "mean": sum(sample) / len(sample),
        "p50": percentile(sample, 50),
        "p95": percentile(sample, 95),
        "p99": percentile(sample, 99),
        "max": max(sample),
    }


def _proc_table() -> dict[int, dict[str, int]]:
    table: dict[int, dict[str, int]] = {}
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            pid = int(entry.name)
            table[pid] = {
                "pid": pid,
                "ppid": int(fields[1]),
                "cpu_ticks": int(fields[11]) + int(fields[12]),
                "start_ticks": int(fields[19]),
                "rss_bytes": int(
                    (entry / "statm").read_text(encoding="utf-8").split()[1]
                )
                * page_size,
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return table


def _descendants(root_pid: int, table: dict[int, dict[str, int]]) -> set[int]:
    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, record in table.items():
            if record["ppid"] in found and pid not in found:
                found.add(pid)
                changed = True
    return found


class ProcessTreeCpuMeter:
    """Samples every process in a service tree and retains exited descendants."""

    def __init__(self, root_pid: int, interval: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self.root_pid = root_pid
        self.interval = interval
        self.ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        self.baseline: dict[tuple[int, int], int] = {}
        self.maximum: dict[tuple[int, int], dict[str, int]] = {}
        self.max_process_count = 0
        self.max_rss_bytes = 0
        self.samples = 0
        self.started_ns: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self, *, baseline: bool = False) -> None:
        table = _proc_table()
        pids = _descendants(self.root_pid, table)
        records = [table[pid] for pid in pids if pid in table]
        self.samples += 1
        self.max_process_count = max(self.max_process_count, len(records))
        self.max_rss_bytes = max(
            self.max_rss_bytes, sum(record["rss_bytes"] for record in records)
        )
        for record in records:
            identity = (record["pid"], record["start_ticks"])
            if baseline:
                self.baseline[identity] = record["cpu_ticks"]
            previous = self.maximum.get(identity)
            if previous is None or record["cpu_ticks"] > previous["cpu_ticks"]:
                self.maximum[identity] = record

    def start(self) -> None:
        if self.started_ns is not None:
            raise RuntimeError("process-tree CPU meter already started")
        self._sample(baseline=True)
        self.started_ns = time.perf_counter_ns()

        def run() -> None:
            while not self._stop.wait(self.interval):
                self._sample()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self.started_ns is None:
            raise RuntimeError("process-tree CPU meter was not started")
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        stopped_ns = time.perf_counter_ns()
        process_records = []
        total_ticks = 0
        for identity, record in sorted(self.maximum.items()):
            baseline_ticks = self.baseline.get(identity, 0)
            delta = max(0, record["cpu_ticks"] - baseline_ticks)
            total_ticks += delta
            process_records.append(
                {
                    "pid": identity[0],
                    "start_ticks": identity[1],
                    "ppid_last_seen": record["ppid"],
                    "baseline_cpu_ticks": baseline_ticks,
                    "maximum_cpu_ticks": record["cpu_ticks"],
                    "measured_cpu_ticks": delta,
                }
            )
        return {
            "method": "sampled-/proc whole-process-tree utime+stime",
            "root_pid": self.root_pid,
            "sample_interval_ms": self.interval * 1000,
            "sample_count": self.samples,
            "observed_process_identities": len(process_records),
            "max_live_process_count": self.max_process_count,
            "max_tree_rss_bytes": self.max_rss_bytes,
            "cpu_ticks": total_ticks,
            "cpu_seconds": total_ticks / self.ticks_per_second,
            "wall_duration_ns": stopped_ns - self.started_ns,
            "processes": process_records,
        }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(base_url: str, process: subprocess.Popen[Any], timeout: int) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited during startup: {process.returncode}")
        try:
            with opener.open(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"service readiness timed out: {last_error}")


def _verified_revision(model: Path, revision: str) -> bool:
    resolved = model.resolve(strict=True)
    if revision in resolved.parts:
        return True
    marker = resolved / ".cruise-model-revision"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == revision:
        return True
    config = resolved / "config.json"
    if config.is_file():
        payload = json.loads(config.read_text(encoding="utf-8"))
        return payload.get("_commit_hash") == revision
    return False


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


def _write_diagnostic_tail(
    runtime_dir: Path,
    output: Path,
    *,
    max_files: int = 8,
    max_bytes_per_file: int = 8192,
) -> dict[str, Any] | None:
    root = runtime_dir / "cann-logs"
    if not root.is_dir():
        return None
    candidates: list[tuple[int, int, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        candidates.append((stat.st_mtime_ns, stat.st_size, path))
    selected = sorted(candidates, reverse=True)[:max_files]
    if not selected:
        return None
    chunks: list[str] = []
    retained_bytes = 0
    for _, size, path in selected:
        try:
            with path.open("rb") as stream:
                stream.seek(max(0, size - max_bytes_per_file))
                data = stream.read(max_bytes_per_file)
        except OSError:
            continue
        relative = path.relative_to(root)
        chunks.append(f"=== {relative} size={size} tail_bytes={len(data)} ===\n")
        chunks.append(data.decode("utf-8", errors="replace"))
        chunks.append("\n")
        retained_bytes += len(data)
    if retained_bytes == 0:
        return None
    output.write_text("".join(chunks), encoding="utf-8")
    return {
        "path": str(output),
        "source_files": len(selected),
        "retained_source_bytes": retained_bytes,
        "max_files": max_files,
        "max_bytes_per_file": max_bytes_per_file,
    }


def _server_command(args: argparse.Namespace, workload: Workload, port: int) -> list[str]:
    if args.route == "owner":
        return [
            sys.executable,
            "-m",
            "vllm_ascend_persistent_owner.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            workload.served_model_name,
            "--tokenizer",
            str(args.tokenizer),
            "--owner-executable",
            str(args.owner_executable),
            "--function-config",
            str(args.function_config),
            "--graph-config",
            str(args.graph_config),
            "--deploy-config",
            str(args.deploy_config),
            "--air",
            str(args.air),
            "--owner-id",
            str(args.owner_id),
            "--owner-startup-timeout",
            str(OWNER_STARTUP_TIMEOUT_SECONDS),
            "--admission-cohort-size",
            "4",
        ]
    return [
        sys.executable,
        "-m",
        "vllm_ascend_persistent_owner.graph_baseline_launcher",
        "serve",
        str(args.model),
        "--tokenizer",
        str(args.tokenizer),
        "--served-model-name",
        workload.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "384",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "512",
        "--gpu-memory-utilization",
        "0.35",
        "--kv-cache-memory-bytes",
        str(512 * 1024 * 1024),
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--worker-cls",
        "vllm_ascend.worker.worker.NPUWorker",
        "--additional-config",
        GRAPH_ASCEND_ADDITIONAL_CONFIG,
        "--no-async-scheduling",
    ]


def _request_body(workload: Workload, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": workload.served_model_name,
        "prompt": list(spec["prompt_token_ids"]),
        "max_tokens": int(spec["output_tokens"]),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_tokens": 0,
        "ignore_eos": bool(spec["ignore_eos"]),
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def _request(
    client: Any, base_url: str, workload: Workload, spec: dict[str, Any], index: int
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    tokens: list[int] = []
    text = ""
    token_times: list[int] = []
    chunk_sizes: list[int] = []
    finish_reason = None
    usage = None
    done = False
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/completions", json=_request_body(workload, spec)
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
                        raise ValueError("P5 requires choice index zero")
                    new_tokens = list(choice.get("token_ids") or [])
                    now = time.perf_counter_ns()
                    tokens.extend(new_tokens)
                    text += str(choice.get("text") or "")
                    token_times.extend(now for _ in new_tokens)
                    if new_tokens:
                        chunk_sizes.append(len(new_tokens))
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
    except Exception as exc:
        return {
            "request_index": index,
            "tokens": tokens,
            "text": text,
            "finish_reason": finish_reason,
            "done": done,
            "error": f"{type(exc).__name__}: {exc}",
            "pass": False,
        }
    ended = time.perf_counter_ns()
    tpot_ms = (
        (token_times[-1] - token_times[0]) / (len(token_times) - 1) / 1_000_000
        if len(token_times) > 1
        else None
    )
    expected_usage = {
        "prompt_tokens": len(spec["prompt_token_ids"]),
        "completion_tokens": int(spec["output_tokens"]),
        "total_tokens": len(spec["prompt_token_ids"]) + int(spec["output_tokens"]),
    }
    checks = {
        "exact_output_length": len(tokens) == int(spec["output_tokens"]),
        "length_finish": finish_reason == "length",
        "done_boundary": done,
        "single_token_chunks": all(size == 1 for size in chunk_sizes),
        "usage_exact": isinstance(usage, dict)
        and all(usage.get(key) == value for key, value in expected_usage.items()),
    }
    return {
        "request_index": index,
        "tokens": tokens,
        "text": text,
        "finish_reason": finish_reason,
        "usage": usage,
        "done": done,
        "latency_ms": (ended - started) / 1_000_000,
        "ttft_ms": (
            (token_times[0] - started) / 1_000_000 if token_times else None
        ),
        "tpot_ms": tpot_ms,
        "inter_token_ms": [
            (right - left) / 1_000_000
            for left, right in zip(token_times, token_times[1:])
        ],
        "stream_chunk_sizes": chunk_sizes,
        "checks": checks,
        "pass": all(checks.values()),
    }


async def _execute(
    client: Any, base_url: str, workload: Workload, spec: dict[str, Any]
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(int(spec["concurrency"]))

    async def limited(index: int) -> dict[str, Any]:
        async with semaphore:
            return await _request(client, base_url, workload, spec, index)

    results = await asyncio.gather(
        *(limited(index) for index in range(int(spec["request_count"])))
    )
    return sorted(results, key=lambda record: record["request_index"])


async def _owner_metrics(client: Any, base_url: str) -> dict[str, Any] | None:
    try:
        response = await client.get(f"{base_url}/cruise/metrics")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _counter_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    left = before.get("counters", {})
    right = after.get("counters", {})
    return {
        key: int(right.get(key, 0)) - int(left.get(key, 0))
        for key in sorted(set(left) | set(right))
    }


def _metrics(records: list[dict[str, Any]], duration_ns: int, cpu: dict[str, Any]) -> dict[str, Any]:
    output_tokens = sum(len(record.get("tokens", [])) for record in records)
    duration_seconds = duration_ns / 1_000_000_000
    cpu_seconds = float(cpu["cpu_seconds"])
    tpot = [float(record["tpot_ms"]) for record in records if record.get("tpot_ms") is not None]
    return {
        "request_latency_ms": summarize(record["latency_ms"] for record in records),
        "ttft_ms": summarize(record["ttft_ms"] for record in records),
        "tpot_ms": summarize(tpot),
        "inter_token_ms": summarize(
            value for record in records for value in record["inter_token_ms"]
        ),
        "duration_ms": duration_ns / 1_000_000,
        "request_count": len(records),
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / duration_seconds,
        "host_cpu_seconds": cpu_seconds,
        "host_cpu_ms_per_output_token": cpu_seconds * 1000 / output_tokens,
    }


async def _run_load(
    base_url: str, workload: Workload, process: subprocess.Popen[Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, dict[str, int] | None]:
    import httpx

    timeout = httpx.Timeout(3600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        warmup = await _execute(client, base_url, workload, workload.warmup)
        metrics_before = await _owner_metrics(client, base_url)
        meter = ProcessTreeCpuMeter(process.pid)
        meter.start()
        started = time.perf_counter_ns()
        primary = await _execute(client, base_url, workload, workload.primary)
        ended = time.perf_counter_ns()
        cpu = meter.stop()
        cpu["load_duration_ns"] = ended - started
        metrics_after = await _owner_metrics(client, base_url)
    return warmup, primary, cpu, metrics_after, _counter_delta(metrics_before, metrics_after)


def _stop_service(process: subprocess.Popen[Any]) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    assert process.returncode is not None
    return int(process.returncode)


def run_start(args: argparse.Namespace, workload: Workload) -> dict[str, Any]:
    if args.label not in START_ORDER or not args.label.startswith(args.route + "-"):
        raise ValueError("route and label do not match the frozen P5 order")
    if not args.tokenizer.is_dir():
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer}")
    if args.route == "graph":
        if args.model is None or not args.model.is_dir():
            raise FileNotFoundError(f"Graph model not found: {args.model}")
        if not _verified_revision(args.model, workload.revision):
            raise ValueError("Graph model revision is not independently pinned")
    else:
        for path in (
            args.owner_executable,
            args.function_config,
            args.graph_config,
            args.deploy_config,
            args.air,
        ):
            if path is None or not path.exists():
                raise FileNotFoundError(f"Owner runtime input not found: {path}")
    args.runtime_dir.mkdir(parents=True, exist_ok=False)
    for child in ("cache", "cann-logs", "tmp"):
        (args.runtime_dir / child).mkdir()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = _server_command(args, workload, port)
    log_path = args.output.with_name(f"service-{args.label}.log")
    logger_command, logger_metadata = _bounded_logger_command(log_path)
    environment = os.environ.copy()
    environment.update(
        {
            "ASCEND_CACHE_PATH": str(args.runtime_dir / "cache"),
            "ASCEND_PROCESS_LOG_PATH": str(args.runtime_dir / "cann-logs"),
            "TORCHINDUCTOR_CACHE_DIR": str(args.runtime_dir / "cache" / "torchinductor"),
            "TRITON_CACHE_DIR": str(args.runtime_dir / "cache" / "triton"),
            "XDG_CACHE_HOME": str(args.runtime_dir / "cache" / "xdg"),
            "TMPDIR": str(args.runtime_dir / "tmp"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    environment.pop("VLLM_ASCEND_RESIDENT_EPOCH_PLUGIN_ENABLE", None)
    if args.route == "graph":
        environment["VLLM_PLUGINS"] = ",".join(GRAPH_PLUGIN_ALLOWLIST)
    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": "P5 independent service start",
        "route": args.route,
        "run_label": args.label,
        "start_uuid": os.urandom(16).hex(),
        "model": workload.model,
        "revision": workload.revision,
        "model_manifest_sha256": workload.model_manifest_sha256,
        "served_model_name": workload.served_model_name,
        "vllm_plugin_allowlist": (
            list(GRAPH_PLUGIN_ALLOWLIST) if args.route == "graph" else None
        ),
        "graph_rmsnorm_backend": (
            GRAPH_RMSNORM_BACKEND if args.route == "graph" else None
        ),
        "command": command,
        "log": str(log_path),
        "log_metadata": str(logger_metadata),
        "warmup": [],
        "primary": [],
        "pass": False,
    }
    process: subprocess.Popen[Any] | None = None
    logger_process: subprocess.Popen[Any] | None = None
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        logger_process = subprocess.Popen(logger_command, stdin=subprocess.PIPE)
        assert logger_process.stdin is not None
        initialized = time.perf_counter_ns()
        process = subprocess.Popen(
            command,
            stdout=logger_process.stdin,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        logger_process.stdin.close()
        _wait_ready(base_url, process, SERVICE_READINESS_TIMEOUT_SECONDS)
        result["initialization_ms"] = (time.perf_counter_ns() - initialized) / 1_000_000
        warmup, primary, cpu, owner_metrics, owner_delta = asyncio.run(
            _run_load(base_url, workload, process)
        )
        duration_ns = int(cpu["load_duration_ns"])
        result.update(
            {
                "warmup": warmup,
                "primary": primary,
                "process_tree_cpu": cpu,
                "owner_metrics": owner_metrics,
                "owner_counter_delta": owner_delta,
            }
        )
        result["metrics"] = _metrics(primary, duration_ns, cpu)
        route_checks = {
            "legacy_route_not_loaded": args.route != "owner"
            or owner_metrics is not None
            and owner_metrics.get("forbidden_runtime_modules") == [],
            "persistent_owner_identity": args.route != "owner"
            or owner_metrics is not None
            and owner_metrics.get("route") == "persistent_device_model_owner",
            "no_host_decode_steps": args.route != "owner"
            or owner_delta is not None
            and owner_delta.get("host_decode_steps") == 0,
            "all_tokens_from_owner": args.route != "owner"
            or owner_delta is not None
            and owner_delta.get("commit_events") == 8192,
            "request_boundary_feeds_only": args.route != "owner"
            or owner_delta is not None
            and owner_delta.get("admission_events") == 32
            and owner_delta.get("credit_events") == 0
            and owner_delta.get("cancel_events") == 0,
            "exact_c4_admission_cohorts": args.route != "owner"
            or owner_delta is not None
            and owner_delta.get("admission_cohorts") == 8
            and owner_delta.get("partial_admission_cohorts") == 0,
            "device_c4_quantum_coverage": args.route != "owner"
            or owner_delta is not None
            and owner_delta.get("aicore_calls") == 3064,
        }
        checks = {
            "warmup_passed": len(warmup) == 4 and all(item["pass"] for item in warmup),
            "primary_passed": len(primary) == 32 and all(item["pass"] for item in primary),
            "whole_process_tree_measured": cpu["observed_process_identities"] >= 2
            and cpu["cpu_seconds"] > 0
            and cpu["sample_count"] >= 2
            and cpu["wall_duration_ns"] >= cpu["load_duration_ns"] > 0,
            **route_checks,
        }
        result["checks"] = checks
        result["pass"] = all(checks.values())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if process is not None:
            try:
                result["service_returncode"] = _stop_service(process)
            except Exception as exc:
                result["shutdown_error"] = f"{type(exc).__name__}: {exc}"
                result["pass"] = False
        if logger_process is not None:
            try:
                result["logger_returncode"] = logger_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                logger_process.kill()
                result["logger_returncode"] = logger_process.wait(timeout=10)
                result["pass"] = False
        if not result["pass"]:
            diagnostic = _write_diagnostic_tail(
                args.runtime_dir,
                args.output.with_name(f"diagnostic-{args.label}.log"),
            )
            if diagnostic is not None:
                result["diagnostic_tail"] = diagnostic
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--owner-executable", type=Path)
    parser.add_argument("--function-config", type=Path)
    parser.add_argument("--graph-config", type=Path)
    parser.add_argument("--deploy-config", type=Path)
    parser.add_argument("--air", type=Path)
    parser.add_argument("--owner-id", type=int, default=5001)
    args = parser.parse_args()
    try:
        result = run_start(args, load_workload(args.workload.resolve(strict=True)))
    except Exception as exc:
        result = {
            "schema_version": 1,
            "gate": "P5 independent service start",
            "route": args.route,
            "run_label": args.label,
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    write_result(args.output, result)
    return 0 if result.get("pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
