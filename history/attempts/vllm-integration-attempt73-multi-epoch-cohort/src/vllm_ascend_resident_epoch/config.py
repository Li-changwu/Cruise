import os
from dataclasses import dataclass


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return value


@dataclass(frozen=True)
class ResidentEpochConfig:
    max_steps: int = 8
    logical_capacity: int = 8
    physical_blocks: int = 8
    blocks_per_request: int = 2
    block_size: int = 128
    max_batch_size: int = 4

    @classmethod
    def from_env(cls) -> "ResidentEpochConfig":
        config = cls(
            max_steps=_read_int("VLLM_ASCEND_RESIDENT_EPOCH_STEPS", 8),
            logical_capacity=_read_int(
                "VLLM_ASCEND_RESIDENT_EPOCH_CAPACITY", 8
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_steps not in (1, 2, 4, 8):
            raise ValueError("resident epoch steps must be one of 1, 2, 4, 8")
        if self.logical_capacity < self.max_steps:
            raise ValueError("logical capacity must be at least max_steps")
        if self.max_batch_size != 4:
            raise ValueError("the current native graph has a fixed maximum B=4")

