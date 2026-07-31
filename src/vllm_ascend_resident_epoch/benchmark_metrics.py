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
        self.counters: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()
        self.epoch_steps: Counter[int] = Counter()
        self._registered = False

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
            return
        self.counters["device_plan_calls"] += 1
        self.counters["device_planned_requests"] += len(plan.requests)
        self.counters["device_planned_model_calls"] += plan.max_steps
        self.epoch_steps[int(plan.max_steps)] += 1

    def record_result(self, result: Any) -> None:
        if self.path is None:
            return
        if result.route != "device":
            self.counters["non_device_results"] += 1
            return
        self.counters["device_epochs"] += 1
        self.counters["device_model_calls"] += int(result.model_calls)
        self.counters["device_request_tokens"] += sum(
            int(value) for value in result.computed_steps.values()
        )
        self.counters["feed_calls"] += int(result.feed_calls)
        self.counters["fetch_calls"] += int(result.fetch_calls)
        self.counters["native_wall_us"] += int(result.wall_us)
        self.counters["native_cpu_us"] += int(result.native_cpu_us)
        self.counters["socket_send_calls"] += int(result.socket_send_calls)
        self.counters["socket_receive_calls"] += int(result.socket_receive_calls)
        if result.kv_imported:
            self.counters["kv_imports"] += 1

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
