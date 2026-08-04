"""Source-runner bootstrap for per-process CANN dynamic profiling."""

import os
from pathlib import Path
import sys


if (
    os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_DYNAMIC_PROFILING") == "1"
    or os.getenv("PROFILING_MODE") == "dynamic"
):
    source_root = Path(__file__).resolve().parent / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from vllm_ascend_resident_epoch.dynamic_profiling import (
        configure_dynamic_profiling,
    )

    configure_dynamic_profiling()
