from __future__ import annotations

import atexit
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any


METRICS_ENV = "VLLM_ASCEND_RESIDENT_EPOCH_BENCHMARK_METRICS_PATH"


def _validated_metrics_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{METRICS_ENV} must be an absolute path")
    resolved = path.resolve(strict=False)
    shared_memory = Path("/dev/shm").resolve(strict=False)
    if resolved == shared_memory or shared_memory not in resolved.parents:
        raise ValueError(f"{METRICS_ENV} must be under /dev/shm")
    return resolved


class ResidentEpochBenchmarkMetrics:
    """Low-overhead counters enabled only by the M4a experiment runner."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._event_path = (
            path.with_name(f"{path.stem}.events.jsonl") if path is not None else None
        )
        self._event_fd = (
            os.open(
                self._event_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            if self._event_path is not None
            else None
        )
        self.counters: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()
        self.epoch_steps: Counter[int] = Counter()
        self._registered = False

    def _append_event(self, event: dict[str, Any]) -> None:
        if self._event_fd is None:
            return
        os.write(self._event_fd, (json.dumps(event, sort_keys=True) + "\n").encode())

    @classmethod
    def from_env(cls) -> "ResidentEpochBenchmarkMetrics":
        raw = os.getenv(METRICS_ENV)
        metrics = cls(_validated_metrics_path(raw) if raw else None)
        if metrics.path is not None:
            atexit.register(metrics.flush)
            metrics._registered = True
        return metrics

    def record_schedule(self, plan: Any | None, rejection: str | None) -> None:
        if self.path is None:
            return
        self.counters["schedule_calls"] += 1
        if plan is None:
            self.counters["host_schedule_calls"] += 1
            self.rejections[rejection or "unspecified"] += 1
            self._append_event(
                {"kind": "schedule", "device": False, "rejection": rejection}
            )
        else:
            self.counters["device_plan_calls"] += 1
            self.counters["device_planned_requests"] += len(plan.requests)
            self.counters["device_planned_model_calls"] += plan.max_steps
            self.epoch_steps[int(plan.max_steps)] += 1
            self._append_event(
                {
                    "kind": "schedule",
                    "device": True,
                    "request_count": len(plan.requests),
                    "max_steps": int(plan.max_steps),
                }
            )

    def record_result(self, result: Any) -> None:
        if self.path is None:
            return
        if result.route != "device":
            self.counters["non_device_results"] += 1
            self._append_event({"kind": "result", "route": result.route})
        else:
            computed_steps = {
                str(key): int(value) for key, value in result.computed_steps.items()
            }
            self.counters["device_epochs"] += 1
            self.counters["device_model_calls"] += int(result.model_calls)
            self.counters["device_request_tokens"] += sum(computed_steps.values())
            self.counters["feed_calls"] += int(result.feed_calls)
            self.counters["fetch_calls"] += int(result.fetch_calls)
            self.counters["native_wall_us"] += int(result.wall_us)
            self.counters["native_cpu_us"] += int(result.native_cpu_us)
            self.counters["socket_send_calls"] += int(result.socket_send_calls)
            self.counters["socket_receive_calls"] += int(result.socket_receive_calls)
            if result.kv_imported:
                self.counters["kv_imports"] += 1
            self._append_event(
                {
                    "kind": "result",
                    "route": "device",
                    "model_calls": int(result.model_calls),
                    "computed_steps": computed_steps,
                    "feed_calls": int(result.feed_calls),
                    "fetch_calls": int(result.fetch_calls),
                    "wall_us": int(result.wall_us),
                    "native_cpu_us": int(result.native_cpu_us),
                    "socket_send_calls": int(result.socket_send_calls),
                    "socket_receive_calls": int(result.socket_receive_calls),
                    "kv_imported": bool(result.kv_imported),
                }
            )

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "Cruise M4a benchmark-only resident epoch counters",
            "counters": dict(sorted(self.counters.items())),
            "epoch_steps": {
                str(key): value for key, value in sorted(self.epoch_steps.items())
            },
            "rejections": dict(sorted(self.rejections.items())),
        }

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.as_record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        if self._event_fd is not None:
            os.close(self._event_fd)
            self._event_fd = None


def replay_event_journal(path: Path) -> dict[str, Any]:
    """Reconstruct counters after an EngineCore exits without running atexit."""
    metrics = ResidentEpochBenchmarkMetrics()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.get("kind")
            if kind == "schedule":
                if event.get("device"):
                    metrics.counters["device_plan_calls"] += 1
                    metrics.counters["device_planned_requests"] += int(
                        event["request_count"]
                    )
                    metrics.counters["device_planned_model_calls"] += int(
                        event["max_steps"]
                    )
                    metrics.epoch_steps[int(event["max_steps"])] += 1
                else:
                    metrics.counters["host_schedule_calls"] += 1
                    metrics.rejections[event.get("rejection") or "unspecified"] += 1
                metrics.counters["schedule_calls"] += 1
            elif kind == "result":
                if event.get("route") != "device":
                    metrics.counters["non_device_results"] += 1
                    continue
                metrics.counters["device_epochs"] += 1
                metrics.counters["device_model_calls"] += int(event["model_calls"])
                metrics.counters["device_request_tokens"] += sum(
                    int(value) for value in event["computed_steps"].values()
                )
                for name in (
                    "feed_calls",
                    "fetch_calls",
                    "wall_us",
                    "native_cpu_us",
                    "socket_send_calls",
                    "socket_receive_calls",
                ):
                    counter = "native_wall_us" if name == "wall_us" else name
                    metrics.counters[counter] += int(event[name])
                if event.get("kv_imported"):
                    metrics.counters["kv_imports"] += 1
    return metrics.as_record()
