from __future__ import annotations

import importlib


def ensure_triton_ascend_runtime() -> str:
    """Expose the legacy Ascend namespace when Triton uses the CANN name."""
    extra = importlib.import_module("triton.language.extra")
    try:
        native = extra.ascend
        native.libdevice.pow
    except AttributeError:
        cann = importlib.import_module("triton.language.extra.cann")
        cann.libdevice.pow
        extra.ascend = cann
        return "triton.language.extra.cann.libdevice.pow"
    return "triton.language.extra.ascend.libdevice.pow"
