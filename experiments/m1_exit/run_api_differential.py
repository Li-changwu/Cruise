#!/usr/bin/env python3
"""Run the M1 OpenAI-compatible HTTP API differential gate."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from experiments.m1_batched_prefill.run_differential import (
    SCHEDULER_QUALNAME,
    WORKER_QUALNAME,
    write_result,
)


@dataclass(frozen=True)
class ApiCase:
    name: str
    stream: bool
    prompt: list[int] | list[list[int]]
    max_tokens: int
    min_tokens: int | None = None
    stop_token_ids: tuple[int, ...] | None = None
    disconnect_after_tokens: int | None = None

    @property
    def choice_count(self) -> int:
        if self.prompt and isinstance(self.prompt[0], list):
            return len(self.prompt)
        return 1

    @property
    def expected_token_count(self) -> int:
        return 2 if self.stop_token_ids is not None else self.max_tokens

    @property
    def expected_finish_reason(self) -> str:
        return "stop" if self.stop_token_ids is not None else "length"


@dataclass(frozen=True)
class ApiManifest:
    served_model_name: str
    tokenizer: Path
    cases: tuple[ApiCase, ...]


def load_manifest(path: Path) -> ApiManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported M1 API manifest schema")
    model_name = payload.get("served_model_name")
    tokenizer = payload.get("tokenizer")
    raw_cases = payload.get("cases")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("served_model_name must be non-empty")
    if not isinstance(tokenizer, str) or not tokenizer:
        raise ValueError("tokenizer must be a path")
    if not isinstance(raw_cases, list) or len(raw_cases) != 8:
        raise ValueError("the M1 API gate requires eight cases")

    cases: list[ApiCase] = []
    names: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("each API case must be an object")
        name = raw.get("name")
        stream = raw.get("stream")
        prompt = raw.get("prompt")
        max_tokens = raw.get("max_tokens")
        min_tokens = raw.get("min_tokens")
        stop_token_ids = raw.get("stop_token_ids")
        disconnect_after = raw.get("disconnect_after_tokens")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("API case names must be unique")
        names.add(name)
        if not isinstance(stream, bool):
            raise ValueError(f"{name}: stream must be boolean")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"{name}: prompt must be a non-empty token list")
        prompts = prompt if isinstance(prompt[0], list) else [prompt]
        if not all(
            isinstance(tokens, list)
            and 2 <= len(tokens) <= 5
            and all(isinstance(token, int) and token >= 0 for token in tokens)
            for tokens in prompts
        ):
            raise ValueError(f"{name}: invalid prompt tokens")
        if not isinstance(max_tokens, int) or max_tokens < 2:
            raise ValueError(f"{name}: max_tokens must be at least two")
        if any(len(tokens) + max_tokens - 1 > 8 for tokens in prompts):
            raise ValueError(f"{name}: request exceeds resident capacity")
        if min_tokens is not None and (
            not isinstance(min_tokens, int) or min_tokens < 1
        ):
            raise ValueError(f"{name}: invalid min_tokens")
        if stop_token_ids is not None and (
            not isinstance(stop_token_ids, list)
            or not stop_token_ids
            or not all(isinstance(token, int) and token >= 0 for token in stop_token_ids)
        ):
            raise ValueError(f"{name}: invalid stop_token_ids")
        if disconnect_after is not None and (
            not stream
            or not isinstance(disconnect_after, int)
            or not 1 <= disconnect_after < max_tokens
        ):
            raise ValueError(f"{name}: invalid disconnect boundary")
        cases.append(
            ApiCase(
                name=name,
                stream=stream,
                prompt=prompt,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                stop_token_ids=(
                    tuple(stop_token_ids) if stop_token_ids is not None else None
                ),
                disconnect_after_tokens=disconnect_after,
            )
        )
    required = {
        "nonstream-single",
        "nonstream-batch",
        "stream-single",
        "stream-batch",
        "eos-second-token",
        "unsupported-min-tokens",
        "disconnect-after-device",
        "post-disconnect-probe",
    }
    if names != required:
        raise ValueError("the M1 API case set is incomplete")
    return ApiManifest(model_name, Path(tokenizer), tuple(cases))


def _with_tokenizer_override(manifest: ApiManifest) -> ApiManifest:
    override = os.environ.get("CRUISE_API_TOKENIZER")
    if override is None:
        return manifest
    return ApiManifest(manifest.served_model_name, Path(override), manifest.cases)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(base_url: str, process: subprocess.Popen[Any], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"API server exited during startup: {process.returncode}"
            )
        try:
            with urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
                last_error = f"health returned {response.status}"
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"API server readiness timed out: {last_error}")


def _choice_record(choice: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": choice["index"],
        "token_ids": list(choice.get("token_ids") or []),
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
    }


async def _nonstream_case(
    client: Any, base_url: str, model_name: str, case: ApiCase
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_name,
        "prompt": case.prompt,
        "max_tokens": case.max_tokens,
        "temperature": 0.0,
        "return_token_ids": True,
    }
    if case.min_tokens is not None:
        body["min_tokens"] = case.min_tokens
    if case.stop_token_ids is not None:
        body["stop_token_ids"] = list(case.stop_token_ids)
    response = await client.post(f"{base_url}/v1/completions", json=body)
    response.raise_for_status()
    payload = response.json()
    choices = sorted(
        (_choice_record(choice) for choice in payload["choices"]),
        key=lambda choice: choice["index"],
    )
    checks = {
        "choice_count": len(choices) == case.choice_count,
        "choice_indices": [choice["index"] for choice in choices]
        == list(range(case.choice_count)),
        "exact_lengths": all(
            len(choice["token_ids"]) == case.expected_token_count
            for choice in choices
        ),
        "expected_finish": all(
            choice["finish_reason"] == case.expected_finish_reason
            for choice in choices
        ),
        "stop_token_observed": case.stop_token_ids is None
        or all(
            choice["token_ids"][-1] in case.stop_token_ids for choice in choices
        ),
    }
    return {
        "name": case.name,
        "stream": False,
        "choices": choices,
        "usage": payload.get("usage"),
        "checks": checks,
        "pass": all(checks.values()),
    }


async def _stream_case(
    client: Any, base_url: str, model_name: str, case: ApiCase
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_name,
        "prompt": case.prompt,
        "max_tokens": case.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    if case.stop_token_ids is not None:
        body["stop_token_ids"] = list(case.stop_token_ids)
    choices = {
        index: {
            "index": index,
            "token_ids": [],
            "finish_reason": None,
            "stop_reason": None,
        }
        for index in range(case.choice_count)
    }
    usage = None
    done = False
    disconnected = False
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
                record = choices[choice["index"]]
                record["token_ids"].extend(choice.get("token_ids") or [])
                if choice.get("finish_reason") is not None:
                    record["finish_reason"] = choice["finish_reason"]
                    record["stop_reason"] = choice.get("stop_reason")
            if case.disconnect_after_tokens is not None and len(
                choices[0]["token_ids"]
            ) >= case.disconnect_after_tokens:
                disconnected = True
                break

    ordered = [choices[index] for index in range(case.choice_count)]
    if case.disconnect_after_tokens is not None:
        checks = {
            "closed_before_done": disconnected and not done,
            "exact_disconnect_boundary": len(ordered[0]["token_ids"])
            == case.disconnect_after_tokens,
        }
    else:
        checks = {
            "done_received": done,
            "choice_count": len(ordered) == case.choice_count,
            "exact_lengths": all(
                len(choice["token_ids"]) == case.expected_token_count
                for choice in ordered
            ),
            "expected_finish": all(
                choice["finish_reason"] == case.expected_finish_reason
                for choice in ordered
            ),
            "stop_token_observed": case.stop_token_ids is None
            or all(
                choice["token_ids"][-1] in case.stop_token_ids for choice in ordered
            ),
            "usage_received": usage is not None,
        }
    return {
        "name": case.name,
        "stream": True,
        "choices": ordered,
        "usage": usage,
        "done": done,
        "disconnected": disconnected,
        "checks": checks,
        "pass": all(checks.values()),
    }


async def _run_cases(base_url: str, manifest: ApiManifest) -> list[dict[str, Any]]:
    import httpx

    results: list[dict[str, Any]] = []
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for case in manifest.cases:
            if case.stream:
                result = await _stream_case(
                    client, base_url, manifest.served_model_name, case
                )
            else:
                result = await _nonstream_case(
                    client, base_url, manifest.served_model_name, case
                )
            results.append(result)
            if case.disconnect_after_tokens is not None:
                await asyncio.sleep(0.25)
    return results


def _server_command(
    *, mode: str, model: Path, manifest: ApiManifest, port: int
) -> list[str]:
    vllm_executable = Path(sys.executable).with_name("vllm")
    command = [
        str(vllm_executable),
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
        "--worker-cls",
        WORKER_QUALNAME,
        "--no-async-scheduling",
    ]
    if mode == "cruise":
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
            str(4 * 1024 * 1024),
            "--tail-bytes",
            str(4 * 1024 * 1024),
        ],
        metadata,
    )


def _stop_server(process: subprocess.Popen[Any]) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    assert process.returncode is not None
    return int(process.returncode)


def run_api(
    *, mode: str, model: Path, manifest: ApiManifest, output: Path
) -> dict[str, Any]:
    if not manifest.tokenizer.is_dir():
        raise FileNotFoundError(f"tokenizer directory not found: {manifest.tokenizer}")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server_log = output.with_name(f"api-server-{mode}.log")
    logger_command, server_log_metadata = _bounded_logger_command(server_log)
    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": "M1 OpenAI-compatible API semantics",
        "mode": mode,
        "base_url": base_url,
        "server_log": str(server_log),
        "server_log_metadata": str(server_log_metadata),
        "cases": [],
        "pass": False,
    }
    process: subprocess.Popen[Any] | None = None
    logger_process: subprocess.Popen[Any] | None = None
    try:
        server_log.parent.mkdir(parents=True, exist_ok=True)
        logger_process = subprocess.Popen(logger_command, stdin=subprocess.PIPE)
        assert logger_process.stdin is not None
        process = subprocess.Popen(
            _server_command(mode=mode, model=model, manifest=manifest, port=port),
            stdout=logger_process.stdin,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        logger_process.stdin.close()
        _wait_ready(base_url, process, 900)
        result["cases"] = asyncio.run(_run_cases(base_url, manifest))
        result["checks"] = {
            "all_cases_passed": len(result["cases"]) == len(manifest.cases)
            and all(case["pass"] for case in result["cases"]),
            "disconnect_followed_by_successful_probe": next(
                case
                for case in result["cases"]
                if case["name"] == "disconnect-after-device"
            )["pass"]
            and next(
                case
                for case in result["cases"]
                if case["name"] == "post-disconnect-probe"
            )["pass"],
        }
        result["pass"] = all(result["checks"].values())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if process is not None:
            try:
                result["server_returncode"] = _stop_server(process)
            except Exception as exc:
                result["shutdown_error"] = f"{type(exc).__name__}: {exc}"
                result["pass"] = False
        if logger_process is not None:
            try:
                result["logger_returncode"] = logger_process.wait(timeout=30)
                if result["logger_returncode"] != 0:
                    result["pass"] = False
            except Exception as exc:
                logger_process.kill()
                logger_process.wait(timeout=10)
                result["logger_error"] = f"{type(exc).__name__}: {exc}"
                result["pass"] = False
    return result


def _comparable_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": case.get("name"),
        "stream": case.get("stream"),
        "choices": case.get("choices"),
        "usage": case.get("usage"),
        "done": case.get("done"),
        "disconnected": case.get("disconnected"),
    }


def compare_results(baseline_path: Path, cruise_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    cruise = json.loads(cruise_path.read_text(encoding="utf-8"))
    baseline_cases = {
        case["name"]: _comparable_case(case) for case in baseline.get("cases", [])
    }
    cruise_cases = {
        case["name"]: _comparable_case(case) for case in cruise.get("cases", [])
    }
    mismatches = {
        name: {
            "baseline": baseline_cases.get(name),
            "cruise": cruise_cases.get(name),
        }
        for name in sorted(set(baseline_cases) | set(cruise_cases))
        if baseline_cases.get(name) != cruise_cases.get(name)
    }
    checks = {
        "baseline_passed": baseline.get("pass") is True,
        "cruise_passed": cruise.get("pass") is True,
        "same_case_set": set(baseline_cases) == set(cruise_cases),
        "exact_api_semantics": not mismatches,
    }
    return {
        "schema_version": 1,
        "gate": "M1 OpenAI API differential comparison",
        "baseline": str(baseline_path),
        "cruise": str(cruise_path),
        "mismatches": mismatches,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("baseline", "cruise", "compare"), required=True
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cruise", type=Path)
    args = parser.parse_args()

    if args.mode == "compare":
        if args.baseline is None or args.cruise is None:
            parser.error("compare mode requires --baseline and --cruise")
        result = compare_results(
            args.baseline.resolve(strict=True), args.cruise.resolve(strict=True)
        )
    else:
        if args.model is None:
            parser.error(f"{args.mode} mode requires --model")
        result = run_api(
            mode=args.mode,
            model=args.model.resolve(strict=True),
            manifest=_with_tokenizer_override(
                load_manifest(args.cases.resolve(strict=True))
            ),
            output=args.output,
        )
    write_result(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
