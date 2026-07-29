from vllm_ascend_resident_epoch.sidecar_backend import (
    EXECUTE,
    GRAPH_BATCH_SIZE,
    PROTOCOL_VERSION,
    REQUEST,
    REQUEST_MAGIC,
    RESPONSE,
    RESPONSE_MAGIC,
    WARM_UP,
    WARMUP_GENERATION,
)
from vllm_ascend_resident_epoch.contract import EpochCommitState


def test_sidecar_binary_protocol_sizes_and_round_trip():
    assert REQUEST.size == 128
    assert RESPONSE.size == 352

    request = REQUEST.pack(
        REQUEST_MAGIC,
        PROTOCOL_VERSION,
        EXECUTE,
        2,
        4,
        *range(GRAPH_BATCH_SIZE),
        *range(GRAPH_BATCH_SIZE),
        *([1] * GRAPH_BATCH_SIZE),
        *([151645] * GRAPH_BATCH_SIZE),
        *([1, 2, 0, 0]),
    )
    unpacked = REQUEST.unpack(request)
    assert unpacked[:5] == (REQUEST_MAGIC, PROTOCOL_VERSION, EXECUTE, 2, 4)

    response = RESPONSE.pack(
        RESPONSE_MAGIC,
        0,
        0,
        4,
        1,
        1,
        EpochCommitState.COMMITTED,
        0,
        123,
        45,
        260,
        368,
        *([4, 4, 0, 0]),
        *([1, 2, 0, 0]),
        *range(GRAPH_BATCH_SIZE * 8),
    )
    assert RESPONSE.unpack(response)[:7] == (
        RESPONSE_MAGIC,
        0,
        0,
        4,
        1,
        1,
        123,
    )
    assert RESPONSE.unpack(response)[6:12] == (
        EpochCommitState.COMMITTED,
        0,
        123,
        45,
        260,
        368,
    )


def test_warmup_operation_uses_reserved_generation():
    request = REQUEST.pack(
        REQUEST_MAGIC,
        PROTOCOL_VERSION,
        WARM_UP,
        1,
        1,
        *([11690, 0, 0, 0]),
        *([0] * GRAPH_BATCH_SIZE),
        *([1, 0, 0, 0]),
        *([151645] * GRAPH_BATCH_SIZE),
        *([WARMUP_GENERATION, 0, 0, 0]),
    )
    unpacked = REQUEST.unpack(request)
    assert unpacked[:5] == (REQUEST_MAGIC, PROTOCOL_VERSION, WARM_UP, 1, 1)
    assert unpacked[-GRAPH_BATCH_SIZE:] == (WARMUP_GENERATION, 0, 0, 0)
