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


def _read_binary(name: str, default: bool = False) -> bool:
    value = _read_int(name, int(default))
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return bool(value)


@dataclass(frozen=True)
class ResidentEpochConfig:
    max_steps: int = 8
    logical_capacity: int = 8
    physical_blocks: int = 8
    blocks_per_request: int = 2
    block_size: int = 128
    max_batch_size: int = 4
    fixed_epoch_graph: bool = False

    @classmethod
    def from_env(cls) -> "ResidentEpochConfig":
        fixed_epoch_graph = _read_binary("VLLM_ASCEND_RESIDENT_EPOCH_K6")
        config = cls(
            max_steps=_read_int("VLLM_ASCEND_RESIDENT_EPOCH_STEPS", 8),
            logical_capacity=_read_int(
                "VLLM_ASCEND_RESIDENT_EPOCH_CAPACITY", 8
            ),
            physical_blocks=4 if fixed_epoch_graph else 8,
            blocks_per_request=1 if fixed_epoch_graph else 2,
            fixed_epoch_graph=fixed_epoch_graph,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 1 <= self.max_steps <= 8:
            raise ValueError("resident epoch steps must be in [1, 8]")
        if not isinstance(self.fixed_epoch_graph, bool):
            raise ValueError("fixed epoch graph must be a boolean")
        expected_blocks_per_request = 1 if self.fixed_epoch_graph else 2
        expected_physical_blocks = 4 if self.fixed_epoch_graph else 8
        if self.blocks_per_request != expected_blocks_per_request:
            raise ValueError("resident epoch block-table layout disagrees with graph")
        if self.physical_blocks != expected_physical_blocks:
            raise ValueError("resident epoch KV layout disagrees with graph")
        if self.logical_capacity < self.max_steps:
            raise ValueError("logical capacity must be at least max_steps")
        if self.max_batch_size != 4:
            raise ValueError("the current native graph has a fixed maximum B=4")
