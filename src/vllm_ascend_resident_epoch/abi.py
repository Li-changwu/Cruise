from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    element_bytes: int

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.element_bytes


COMMON_INPUTS = (
    TensorSpec("token", (4, 1), 8),
    TensorSpec("position", (4,), 8),
    TensorSpec("sequence_length", (4, 1), 4),
    TensorSpec("slot_mapping", (4,), 4),
    TensorSpec("active_mask", (4,), 4),
    TensorSpec("block_table", (4, 2), 4),
    TensorSpec("tiling", (72,), 1),
    TensorSpec("control", (11,), 4),
)
KEY_CACHE = TensorSpec("key_cache", (28, 8, 128, 4, 128), 2)
VALUE_CACHE = TensorSpec("value_cache", (28, 8, 128, 4, 128), 2)
TOKEN_HISTORY = TensorSpec("token_history", (8, 4), 8)
CONTROL_OUTPUT = TensorSpec("control_output", (28,), 4)

OLD_INPUTS = COMMON_INPUTS[:3] + (KEY_CACHE,) + COMMON_INPUTS[3:6] + (
    VALUE_CACHE,
) + COMMON_INPUTS[6:]
OLD_OUTPUTS = (
    TensorSpec("logits_history", (8, 4, 1, 152064), 4),
    TOKEN_HISTORY,
    KEY_CACHE,
    VALUE_CACHE,
    TensorSpec("final_token", (4, 1), 8),
    TensorSpec("final_position", (4,), 8),
    TensorSpec("final_length", (4, 1), 4),
    TensorSpec("final_slot", (4,), 4),
    TensorSpec("final_active", (4,), 4),
    CONTROL_OUTPUT,
)
NEW_INPUTS = COMMON_INPUTS
NEW_OUTPUTS = (TOKEN_HISTORY, CONTROL_OUTPUT)


def total_bytes(specs: tuple[TensorSpec, ...]) -> int:
    return sum(spec.nbytes for spec in specs)


ABI_BYTES = {
    "old": {
        "input": total_bytes(OLD_INPUTS),
        "output": total_bytes(OLD_OUTPUTS),
    },
    "new": {
        "input": total_bytes(NEW_INPUTS),
        "output": total_bytes(NEW_OUTPUTS),
    },
}

OLD_TOTAL_BYTES = ABI_BYTES["old"]["input"] + ABI_BYTES["old"]["output"]
NEW_TOTAL_BYTES = ABI_BYTES["new"]["input"] + ABI_BYTES["new"]["output"]
TOTAL_REDUCTION_BYTES = OLD_TOTAL_BYTES - NEW_TOTAL_BYTES
KV_ROUND_TRIP_BYTES = 2 * (KEY_CACHE.nbytes + VALUE_CACHE.nbytes)


assert ABI_BYTES["old"] == {"input": 58_720_516, "output": 78_184_928}
assert ABI_BYTES["new"] == {"input": 260, "output": 368}
assert TOTAL_REDUCTION_BYTES == 136_904_816
assert KV_ROUND_TRIP_BYTES == 117_440_512
