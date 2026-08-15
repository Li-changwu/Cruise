"""Launch stock vLLM-Ascend Graph with the image's Triton namespace alias."""

from __future__ import annotations

import importlib

from vllm_ascend_persistent_owner.graph_rmsnorm_compat import (
    BACKEND,
    install_cann9_rmsnorm_compat,
)


def _ensure_triton_ascend_runtime() -> None:
    extra = importlib.import_module("triton.language.extra")
    try:
        native = extra.ascend
        native.libdevice.pow
    except AttributeError:
        cann = importlib.import_module("triton.language.extra.cann")
        cann.libdevice.pow
        extra.ascend = cann


_ensure_triton_ascend_runtime()
_compat_installed = install_cann9_rmsnorm_compat()
print(
    "CRUISE_P5_GRAPH_RMSNORM_COMPAT "
    f"backend={BACKEND} installed={int(_compat_installed)}",
    flush=True,
)


def main() -> int | None:
    from vllm.entrypoints.cli.main import main as vllm_main

    return vllm_main()


if __name__ == "__main__":
    raise SystemExit(main())
