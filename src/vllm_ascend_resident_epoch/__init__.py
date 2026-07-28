"""vLLM-Ascend fixed-batch resident epoch integration."""

from .contract import (
    ResidentEpochPlan,
    ResidentEpochRequest,
    ResidentEpochResult,
)

__all__ = [
    "ResidentEpochPlan",
    "ResidentEpochRequest",
    "ResidentEpochResult",
]
