"""CANN 9 compatibility for the stock vLLM-Ascend Graph baseline."""

from __future__ import annotations

import importlib
from types import ModuleType


BACKEND = "torch_npu.npu_add_rms_norm"
_MARKER = "_cruise_p5_cann9_rmsnorm_compat"


def _custom_add_rms_norm_bias_available() -> bool:
    return False


def install_cann9_rmsnorm_compat(layernorm: ModuleType | None = None) -> bool:
    """Select vLLM-Ascend's supported no-bias RMSNorm fallback.

    The pinned CANN 9.0.0 libopapi does not provide aclnnAddRmsNormBias.  The
    vLLM-Ascend layernorm implementation already carries a torch-npu fallback;
    replacing its module-local capability check leaves all other custom ops and
    Graph settings unchanged.
    """
    if layernorm is None:
        layernorm = importlib.import_module("vllm_ascend.ops.layernorm")
    if getattr(layernorm, _MARKER, False):
        return False
    layernorm.enable_custom_op = _custom_add_rms_norm_bias_available
    setattr(layernorm, _MARKER, True)
    return True
