"""Run vLLM's model registry without the inherited service-profiler hooks."""

from __future__ import annotations

import os


def main() -> None:
    for name in (
        "SERVICE_PROF_CONFIG_PATH",
        "PROFILING_SYMBOLS_PATH",
        "VLLM_ASCEND_RESIDENT_EPOCH_SERVICE_PROFILER_ENABLE",
    ):
        os.environ.pop(name, None)

    from vllm.model_executor.models.registry import _run

    _run()


if __name__ == "__main__":
    main()
