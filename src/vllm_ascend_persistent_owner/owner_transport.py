"""Async request-boundary transport for the model-load-scoped Device Owner."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterable


_OUTPUT_TYPES = {
    1: "admit_ack",
    2: "commit",
    3: "retire_complete",
    4: "retire_cancelled",
    5: "credit_ack",
    6: "quiescent",
    7: "cumulative_ack",
    8: "rejected",
    9: "shutdown",
}
_FORBIDDEN_RUNTIME_MODULES = (
    "vllm_ascend_resident_epoch.scheduler",
    "vllm_ascend_resident_epoch.sidecar_backend",
    "vllm_ascend_resident_epoch.server_launcher",
)
_LIFECYCLE_TYPES = frozenset(
    {
        "P5_OWNER_PHASE",
        "P5_OWNER_READY",
        "P5_OWNER_EXIT",
        "P5_OWNER_PROTOCOL_ERROR",
    }
)
_PROTOCOL_TYPES = _LIFECYCLE_TYPES | {"P4_OUTPUT"}
_MAX_CHILD_LINE_BYTES = 512
_DIAGNOSTIC_TAIL_LINES = 16


@dataclass(frozen=True)
class OwnerEvent:
    kind: str
    request: int
    generation: int
    row: int
    commit_seq: int
    token: int
    position: int
    page: int
    status: int
    aicore_calls: int
    total_commits: int
    total_retired: int
    checksum: int
    finish_reason: int


@dataclass
class OwnerRequest:
    request: int
    generation: int
    prompt_tokens: tuple[int, ...]
    max_tokens: int
    events: asyncio.Queue[OwnerEvent | BaseException] = field(
        default_factory=asyncio.Queue
    )
    row: int = -1
    retired: bool = False
    cancel_seq: int = 0

    @property
    def key(self) -> tuple[int, int]:
        return self.request, self.generation


def _parse_fields(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split()
    if not parts:
        return "", {}
    marker_index = next(
        (index for index, part in enumerate(parts) if part in _PROTOCOL_TYPES), 0
    )
    fields: dict[str, str] = {}
    for part in parts[marker_index + 1 :]:
        key, separator, value = part.partition("=")
        if separator:
            fields[key] = value
    return parts[marker_index], fields


def _format_protocol_line(prefix: str, fields: dict[str, str]) -> str:
    values = " ".join(f"{key}={value}" for key, value in fields.items())
    return f"{prefix} {values}".rstrip()


class OwnerTransport:
    """Owns one native Host transport around one persistent Device model owner.

    The transport sends only admission, credit, cancellation and shutdown
    events. It has no token-step or Decode-position API.
    """

    def __init__(
        self,
        command: Iterable[str | Path],
        *,
        startup_timeout: float = 1200.0,
    ) -> None:
        self.command = tuple(str(item) for item in command)
        self.startup_timeout = startup_timeout
        self.process: asyncio.subprocess.Process | None = None
        self.requests: dict[tuple[int, int], OwnerRequest] = {}
        self._write_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._reader_task: asyncio.Task[None] | None = None
        self._log_tail: deque[str] = deque(maxlen=32)
        self._closed = False
        self._reader_error: BaseException | None = None
        self._counters = {
            "admission_events": 0,
            "admission_cohorts": 0,
            "partial_admission_cohorts": 0,
            "credit_events": 0,
            "cancel_events": 0,
            "shutdown_events": 0,
            "output_events": 0,
            "commit_events": 0,
            "rejected_events": 0,
            "aicore_calls": 0,
            "total_commits": 0,
            "total_retired": 0,
            "host_decode_steps": 0,
        }

    async def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("Persistent Owner transport already started")
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._reader_task = asyncio.create_task(self._read_outputs())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self.startup_timeout)
            if self._reader_error is not None:
                raise self._reader_error
        except BaseException as exc:
            await self._terminate()
            print(
                "P5_OWNER_START_FAILURE "
                f"error={type(exc).__name__} timeout_seconds={self.startup_timeout:g}",
                file=sys.stderr,
                flush=True,
            )
            for line in tuple(self._log_tail)[-_DIAGNOSTIC_TAIL_LINES:]:
                print(
                    f"P5_OWNER_DIAGNOSTIC {line[:_MAX_CHILD_LINE_BYTES]}",
                    file=sys.stderr,
                    flush=True,
                )
            raise RuntimeError(
                "Persistent Owner failed to become ready: "
                + " | ".join(self._log_tail)
            ) from (self._reader_error or exc)

    async def _read_outputs(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                raw = await self.process.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if len(line) > _MAX_CHILD_LINE_BYTES:
                    line = line[:_MAX_CHILD_LINE_BYTES] + "..."
                self._log_tail.append(line)
                prefix, fields = _parse_fields(line)
                if prefix in _LIFECYCLE_TYPES:
                    print(_format_protocol_line(prefix, fields), flush=True)
                if prefix == "P5_OWNER_READY":
                    self._ready.set()
                elif prefix == "P4_OUTPUT":
                    self._dispatch_output(fields)
            returncode = await self.process.wait()
            if not self._closed and (returncode != 0 or not self._ready.is_set()):
                raise RuntimeError(
                    f"Persistent Owner exited with status {returncode} before readiness"
                )
        except BaseException as exc:
            self._reader_error = exc
            self._ready.set()
            for request in self.requests.values():
                if not request.retired:
                    request.events.put_nowait(exc)

    def _dispatch_output(self, raw: dict[str, str]) -> None:
        try:
            output_type = int(raw["type"])
            event = OwnerEvent(
                kind=_OUTPUT_TYPES[output_type],
                request=int(raw["request"]),
                generation=int(raw["generation"]),
                row=int(raw["row"]),
                commit_seq=int(raw["commit_seq"]),
                token=int(raw["token"]),
                position=int(raw["position"]),
                page=int(raw.get("page", "-1")),
                status=int(raw["status"]),
                aicore_calls=int(raw["aicore_calls"]),
                total_commits=int(raw["total_commits"]),
                total_retired=int(raw["total_retired"]),
                checksum=int(raw.get("checksum", "0")),
                finish_reason=int(raw["finish_reason"]),
            )
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"invalid Persistent Owner output: {raw}") from exc
        self._counters["output_events"] += 1
        self._counters["aicore_calls"] = max(
            self._counters["aicore_calls"], event.aicore_calls
        )
        self._counters["total_commits"] = max(
            self._counters["total_commits"], event.total_commits
        )
        self._counters["total_retired"] = max(
            self._counters["total_retired"], event.total_retired
        )
        if event.kind == "commit":
            self._counters["commit_events"] += 1
        elif event.kind == "rejected":
            self._counters["rejected_events"] += 1
        request = self.requests.get((event.request, event.generation))
        if request is None:
            return
        if event.kind == "admit_ack":
            request.row = event.row
        elif event.kind.startswith("retire_"):
            request.retired = True
        request.events.put_nowait(event)

    async def _write(self, commands: list[str]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Persistent Owner is not running")
        if self.process.returncode is not None:
            raise RuntimeError(
                f"Persistent Owner exited with status {self.process.returncode}"
            )
        async with self._write_lock:
            self.process.stdin.write(("\n".join(commands) + "\n").encode("ascii"))
            await self.process.stdin.drain()

    async def admit_many(
        self,
        requests: Iterable[OwnerRequest],
        *,
        cohort_id: int,
        ignore_eos: bool,
        eos_token: int,
    ) -> None:
        batch = tuple(requests)
        if not 1 <= len(batch) <= 4:
            raise ValueError("a Persistent Owner cohort must contain one to four requests")
        commands: list[str] = []
        for request in batch:
            if request.key in self.requests:
                raise ValueError(f"duplicate Persistent Owner request {request.key}")
            self.requests[request.key] = request
            prompt = " ".join(str(token) for token in request.prompt_tokens)
            commands.append(
                "ADMIT "
                f"{request.request} {request.generation} {request.max_tokens} "
                f"{request.max_tokens} {eos_token} {1 if ignore_eos else 0} "
                f"{cohort_id} {len(batch)} {len(request.prompt_tokens)} {prompt}"
            )
        try:
            await self._write(commands)
        except BaseException:
            for request in batch:
                self.requests.pop(request.key, None)
            raise
        self._counters["admission_events"] += len(batch)
        self._counters["admission_cohorts"] += 1
        if len(batch) != 4:
            self._counters["partial_admission_cohorts"] += 1

    async def cancel(self, request: OwnerRequest) -> None:
        if request.retired or request.row < 0:
            return
        request.cancel_seq += 1
        await self._write(
            [
                f"CANCEL {request.request} {request.generation} "
                f"{request.row} {request.cancel_seq}"
            ]
        )
        self._counters["cancel_events"] += 1

    def metrics(self) -> dict[str, Any]:
        forbidden = sorted(name for name in _FORBIDDEN_RUNTIME_MODULES if name in sys.modules)
        return {
            "schema_version": 1,
            "route": "persistent_device_model_owner",
            "owner_lifetime": "model-load-to-model-unload",
            "host_visible_decode_epoch": False,
            "host_token_step_api": False,
            "async_output_drain": True,
            "native_pid": self.process.pid if self.process is not None else None,
            "forbidden_runtime_modules": forbidden,
            "counters": dict(self._counters),
            "live_requests": sum(not item.retired for item in self.requests.values()),
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process is None:
            return
        for request in tuple(self.requests.values()):
            if not request.retired and request.row >= 0:
                try:
                    await self.cancel(request)
                except (BrokenPipeError, ConnectionError, RuntimeError):
                    break
        deadline = asyncio.get_running_loop().time() + 10.0
        while any(not item.retired for item in self.requests.values()):
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0.01)
        if self.process.returncode is None:
            try:
                await self._write(["SHUTDOWN"])
                self._counters["shutdown_events"] += 1
                assert self.process.stdin is not None
                self.process.stdin.close()
                await asyncio.wait_for(self.process.wait(), timeout=180.0)
            except (BrokenPipeError, ConnectionError, RuntimeError, asyncio.TimeoutError):
                await self._terminate()
        if self._reader_task is not None:
            try:
                await self._reader_task
            except BaseException:
                pass

    async def _terminate(self) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()
