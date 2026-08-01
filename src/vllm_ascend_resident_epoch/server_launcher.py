"""Bootstrap the vLLM CLI with Cruise's required Ascend compatibility."""

from __future__ import annotations

from .triton_compat import ensure_triton_ascend_runtime


# Multiprocessing spawn re-executes this module in each EngineCore process.
ensure_triton_ascend_runtime()


def main() -> int | None:
    from vllm.entrypoints.cli.main import main as vllm_main

    return vllm_main()


if __name__ == "__main__":
    raise SystemExit(main())
