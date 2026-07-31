from vllm_ascend_resident_epoch.sidecar_backend import (
    DEVICE_IPC_EXECUTE,
    EXECUTE,
    GRAPH_BATCH_SIZE,
    PROTOCOL_VERSION,
    REQUEST,
    REQUEST_MAGIC,
    RESPONSE,
    RESPONSE_MAGIC,
    WARM_UP,
    WARMUP_GENERATION,
    SidecarDataFlowEngine,
)
from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
    EpochCommitState,
    ResidentEpochPlan,
    ResidentEpochRequest,
)
from vllm_ascend_resident_epoch.kv_transfer import IPC_METADATA_BYTES


def test_sidecar_binary_protocol_sizes_and_round_trip():
    assert REQUEST.size == 136
    assert RESPONSE.size == 352

    request = REQUEST.pack(
        REQUEST_MAGIC,
        PROTOCOL_VERSION,
        EXECUTE,
        2,
        4,
        0,
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
        0x12345678,
        123,
        45,
        260,
        368,
        *([4, 4, 0, 0]),
        *([1, 2, 0, 0]),
        *range(GRAPH_BATCH_SIZE * 8),
    )
    assert RESPONSE.unpack(response)[:6] == (
        RESPONSE_MAGIC,
        0,
        0,
        4,
        1,
        1,
    )
    assert RESPONSE.unpack(response)[6:12] == (
        EpochCommitState.COMMITTED,
        0x12345678,
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
        0,
        *([11690, 0, 0, 0]),
        *([0] * GRAPH_BATCH_SIZE),
        *([1, 0, 0, 0]),
        *([151645] * GRAPH_BATCH_SIZE),
        *([WARMUP_GENERATION, 0, 0, 0]),
    )
    unpacked = REQUEST.unpack(request)
    assert unpacked[:5] == (REQUEST_MAGIC, PROTOCOL_VERSION, WARM_UP, 1, 1)
    assert unpacked[-GRAPH_BATCH_SIZE:] == (WARMUP_GENERATION, 0, 0, 0)


def test_device_ipc_execute_is_reported_as_kv_import():
    class RecordingSocket:
        def __init__(self):
            self.payloads = []

        def sendall(self, payload):
            self.payloads.append(payload)

    engine = SidecarDataFlowEngine.__new__(SidecarDataFlowEngine)
    engine.socket = RecordingSocket()
    response = RESPONSE.unpack(
        RESPONSE.pack(
            RESPONSE_MAGIC,
            0,
            0,
            1,
            1,
            1,
            EpochCommitState.COMMITTED,
            0x12345678,
            123,
            45,
            3776,
            368,
            *([1, 0, 0, 0]),
            *([7, 0, 0, 0]),
            42,
            *([-1] * (GRAPH_BATCH_SIZE * 8 - 1)),
        )
    )
    engine._receive_response = lambda: response
    plan = ResidentEpochPlan(
        version=CONTRACT_VERSION,
        graph_batch_size=4,
        max_steps=1,
        logical_capacity=8,
        requests=(
            ResidentEpochRequest(
                req_id="request-0",
                row=0,
                generation=7,
                token_id=11,
                position=1,
                sequence_length=2,
                eos_token_id=151645,
                scheduler_block_ids=(0,),
                device_block_ids=(0, 1),
                kv_import_required=True,
            ),
        ),
        active_mask=(1, 0, 0, 0),
    )

    metadata = b"\0" * IPC_METADATA_BYTES
    output = engine._execute(
        plan,
        operation=DEVICE_IPC_EXECUTE,
        transfer_id=9,
        ipc_metadata=metadata,
    )

    assert output.kv_imported is True
    assert output.kv_import_checksum == 0x12345678
    assert len(engine.socket.payloads) == 1
    assert engine.socket.payloads[0][REQUEST.size:] == metadata
