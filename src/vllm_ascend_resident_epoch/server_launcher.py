"""Bootstrap the vLLM CLI with Cruise's required Ascend compatibility."""

from __future__ import annotations

import os
import sys

from .triton_compat import ensure_triton_ascend_runtime
from .dynamic_profiling import (
    configure_dynamic_profiling,
    install_engine_core_dynamic_profiling_rebind,
)


# Multiprocessing spawn re-executes this module in each EngineCore process.
ensure_triton_ascend_runtime()

# Bind the API process now, then install the EngineCore entrypoint wrapper for
# vLLM's normal fork path. The wrapper rebinds after fork before NPU setup.
configure_dynamic_profiling()
install_engine_core_dynamic_profiling_rebind()

# Source-tree benchmark runs have no installed entry-point metadata. Register
# explicitly in every spawned process when the Cruise route opts in.
if os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_PLUGIN_ENABLE") == "1":
    from .plugin import register as register_resident_epoch_plugin

    register_resident_epoch_plugin()

# Service profiling is registered through vLLM's normal general-plugin path in
# both API and EngineCore processes. Its model-registry helper also loads that
# plugin, but cannot safely initialize it. Redirect only that helper to a
# wrapper which clears profiling state before importing vLLM's registry module.
if os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_SERVICE_PROFILER_ENABLE") == "1":
    from vllm.model_executor.models import registry

    registry._SUBPROCESS_COMMAND = [  # type: ignore[attr-defined]
        sys.executable,
        "-m",
        "vllm_ascend_resident_epoch.registry_launcher",
    ]


def main() -> int | None:
    from vllm.entrypoints.cli.main import main as vllm_main

    return vllm_main()


if __name__ == "__main__":
    raise SystemExit(main())
