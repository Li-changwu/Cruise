"""Bootstrap the vLLM CLI with Cruise's required Ascend compatibility."""

from __future__ import annotations

import os

from .triton_compat import ensure_triton_ascend_runtime


# Multiprocessing spawn re-executes this module in each EngineCore process.
ensure_triton_ascend_runtime()

# Source-tree benchmark runs have no installed entry-point metadata. Register
# explicitly in every spawned process when the Cruise route opts in.
if os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_PLUGIN_ENABLE") == "1":
    from .plugin import register as register_resident_epoch_plugin

    register_resident_epoch_plugin()


def main() -> int | None:
    from vllm.entrypoints.cli.main import main as vllm_main

    return vllm_main()


if __name__ == "__main__":
    raise SystemExit(main())
