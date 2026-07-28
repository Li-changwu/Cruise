import ctypes
import os
from pathlib import Path

from .backend import NativeEpochOutput
from .contract import ResidentEpochPlan


MAX_EPOCH_STEPS = 8


def _required_path(name: str, *, directory: bool = False) -> Path:
    raw = os.getenv(name)
    if not raw:
        raise RuntimeError(f"{name} is required by the DataFlow backend")
    path = Path(raw).resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise RuntimeError(f"{name} must name an existing {kind}: {path}")
    return path


class CtypesDataFlowEngine:
    def __init__(self) -> None:
        library_path = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_LIBRARY")
        air_path = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_AIR")
        graph_config = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_GRAPH_CONFIG")
        func_config = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_FUNC_CONFIG")
        tiling_path = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_TILING")
        external_weights = _required_path(
            "VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS", directory=True
        )
        if not str(external_weights).startswith("/dev/shm/"):
            raise RuntimeError("resident epoch external weights must be in /dev/shm")

        self.library = ctypes.CDLL(str(library_path))
        self.library.resident_epoch_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.library.resident_epoch_create.restype = ctypes.c_void_p
        self.library.resident_epoch_execute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.library.resident_epoch_execute.restype = ctypes.c_int32
        self.library.resident_epoch_destroy.argtypes = [ctypes.c_void_p]
        self.library.resident_epoch_destroy.restype = None

        create_status = ctypes.c_int32(-1)
        self.handle = self.library.resident_epoch_create(
            str(air_path).encode(),
            str(graph_config).encode(),
            str(func_config).encode(),
            str(external_weights).encode(),
            str(tiling_path).encode(),
            ctypes.byref(create_status),
        )
        if not self.handle or create_status.value != 0:
            raise RuntimeError(
                f"failed to create DataFlow resident epoch, status={create_status.value}"
            )

    def execute(self, plan: ResidentEpochPlan) -> NativeEpochOutput:
        if plan.graph_batch_size != 4:
            raise ValueError("the ctypes backend owns only the static B=4 graph")
        request_count = len(plan.requests)
        token_ids = (ctypes.c_int64 * request_count)(
            *(request.token_id for request in plan.requests)
        )
        positions = (ctypes.c_int64 * request_count)(
            *(request.position for request in plan.requests)
        )
        sequence_lengths = (ctypes.c_int32 * request_count)(
            *(request.sequence_length for request in plan.requests)
        )
        eos_token_ids = (ctypes.c_int32 * request_count)(
            *(request.eos_token_id for request in plan.requests)
        )
        output_tokens = (ctypes.c_int64 * (request_count * MAX_EPOCH_STEPS))()
        output_executed = (ctypes.c_int32 * request_count)()
        model_calls = ctypes.c_int32(0)
        device_status = ctypes.c_int32(-1)
        feed_calls = ctypes.c_int32(0)
        fetch_calls = ctypes.c_int32(0)
        wall_us = ctypes.c_int64(0)

        status = self.library.resident_epoch_execute(
            self.handle,
            request_count,
            plan.max_steps,
            token_ids,
            positions,
            sequence_lengths,
            eos_token_ids,
            output_tokens,
            output_executed,
            ctypes.byref(model_calls),
            ctypes.byref(device_status),
            ctypes.byref(feed_calls),
            ctypes.byref(fetch_calls),
            ctypes.byref(wall_us),
        )
        if status != 0:
            raise RuntimeError(f"resident_epoch_execute failed with status {status}")

        tokens_by_req: dict[str, list[int]] = {}
        for row, request in enumerate(plan.requests):
            executed = output_executed[row]
            if executed < 0 or executed > plan.max_steps:
                raise RuntimeError("native backend returned an invalid executed count")
            offset = row * MAX_EPOCH_STEPS
            tokens_by_req[request.req_id] = [
                int(output_tokens[offset + step]) for step in range(executed)
            ]
        return NativeEpochOutput(
            status=device_status.value,
            model_calls=model_calls.value,
            token_ids=tokens_by_req,
            feed_calls=feed_calls.value,
            fetch_calls=fetch_calls.value,
            wall_us=wall_us.value,
        )

    def close(self) -> None:
        if self.handle:
            self.library.resident_epoch_destroy(self.handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def create_ctypes_engine() -> CtypesDataFlowEngine:
    return CtypesDataFlowEngine()

