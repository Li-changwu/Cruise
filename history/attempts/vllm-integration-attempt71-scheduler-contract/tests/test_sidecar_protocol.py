from vllm_ascend_resident_epoch.sidecar_backend import (
    EXECUTE,
    GRAPH_BATCH_SIZE,
    PROTOCOL_VERSION,
    REQUEST,
    REQUEST_MAGIC,
    RESPONSE,
    RESPONSE_MAGIC,
)


def test_sidecar_binary_protocol_sizes_and_round_trip():
    assert REQUEST.size == 112
    assert RESPONSE.size == 304

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
        123,
        *([4, 4, 0, 0]),
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
