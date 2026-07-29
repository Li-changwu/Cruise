"""vLLM-Ascend fixed-batch resident epoch integration."""

from .contract import (
    EpochCommitState,
    ResidentEpochExecutionError,
    ResidentEpochPlan,
    ResidentEpochRequest,
    ResidentEpochResult,
)
from .version import PACKAGE_VERSION

__version__ = PACKAGE_VERSION

__all__ = [
    "ResidentEpochPlan",
    "ResidentEpochRequest",
    "ResidentEpochResult",
    "EpochCommitState",
    "ResidentEpochExecutionError",
    "__version__",
]
