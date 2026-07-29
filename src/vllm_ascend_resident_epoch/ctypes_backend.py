import ctypes
import os
from pathlib import Path

from .backend import NativeEpochOutput
from .contract import (
    EpochCommitState,
    ResidentEpochExecutionError,
    ResidentEpochPlan,
)
from .kv_transfer import ResidentKVSnapshot, write_kv_snapshot


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
            ctypes.c_void_p,  # opaque
            ctypes.c_int32,  # request_count
            ctypes.c_int32,  # max_steps
            ctypes.POINTER(ctypes.c_int64),  # input_token_ids
            ctypes.POINTER(ctypes.c_int64),  # input_positions
            ctypes.POINTER(ctypes.c_int32),  # input_sequence_lengths
            ctypes.POINTER(ctypes.c_int32),  # input_eos_token_ids
            ctypes.POINTER(ctypes.c_int32),  # input_row_generations
            ctypes.POINTER(ctypes.c_int64),  # output_token_ids
            ctypes.POINTER(ctypes.c_int32),  # output_executed
            ctypes.POINTER(ctypes.c_int32),  # output_row_generations
            ctypes.POINTER(ctypes.c_int32),  # output_model_calls
            ctypes.POINTER(ctypes.c_int32),  # output_device_status
            ctypes.POINTER(ctypes.c_int32),  # output_feed_calls
            ctypes.POINTER(ctypes.c_int32),  # output_fetch_calls
            ctypes.POINTER(ctypes.c_int32),  # output_commit_state
            ctypes.POINTER(ctypes.c_int32),  # output_kv_import_checksum
            ctypes.POINTER(ctypes.c_int64),  # output_wall_us
            ctypes.POINTER(ctypes.c_int64),  # output_native_cpu_us
            ctypes.POINTER(ctypes.c_int64),  # output_declared_input_bytes
            ctypes.POINTER(ctypes.c_int64),  # output_declared_output_bytes
            ctypes.c_char_p,  # transfer_path
            ctypes.c_uint64,  # transfer_id
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
        return self._execute(plan, transfer_path=None, transfer_id=0)

    def execute_with_import(
        self, plan: ResidentEpochPlan, snapshot: ResidentKVSnapshot
    ) -> NativeEpochOutput:
        path = Path(
            f"/dev/shm/cruise-kv-transfer-{os.getpid()}-{snapshot.transfer_id}"
        )
        write_kv_snapshot(path, snapshot)
        try:
            return self._execute(
                plan,
                transfer_path=path,
                transfer_id=snapshot.transfer_id,
            )
        finally:
            path.unlink(missing_ok=True)

    def _execute(
        self,
        plan: ResidentEpochPlan,
        *,
        transfer_path: Path | None,
        transfer_id: int,
    ) -> NativeEpochOutput:
        if plan.graph_batch_size != 4:
            raise ValueError("the ctypes backend owns only the static B=4 graph")
        request_count = len(plan.requests)
        token_values = [0] * 4
        position_values = [0] * 4
        length_values = [0] * 4
        eos_values = [151645] * 4
        for request in plan.requests:
            token_values[request.row] = request.token_id
            position_values[request.row] = request.position
            length_values[request.row] = request.sequence_length
            eos_values[request.row] = request.eos_token_id
        token_ids = (ctypes.c_int64 * 4)(*token_values)
        positions = (ctypes.c_int64 * 4)(*position_values)
        sequence_lengths = (ctypes.c_int32 * 4)(*length_values)
        eos_token_ids = (ctypes.c_int32 * 4)(*eos_values)
        row_generations = (ctypes.c_int32 * 4)(*plan.row_generations)
        output_tokens = (ctypes.c_int64 * (4 * MAX_EPOCH_STEPS))()
        output_executed = (ctypes.c_int32 * 4)()
        output_row_generations = (ctypes.c_int32 * 4)()
        model_calls = ctypes.c_int32(0)
        device_status = ctypes.c_int32(-1)
        feed_calls = ctypes.c_int32(0)
        fetch_calls = ctypes.c_int32(0)
        commit_state_value = ctypes.c_int32(-1)
        kv_import_checksum = ctypes.c_int32(0)
        wall_us = ctypes.c_int64(0)
        native_cpu_us = ctypes.c_int64(0)
        declared_input_bytes = ctypes.c_int64(0)
        declared_output_bytes = ctypes.c_int64(0)

        status = self.library.resident_epoch_execute(
            self.handle,
            request_count,
            plan.max_steps,
            token_ids,
            positions,
            sequence_lengths,
            eos_token_ids,
            row_generations,
            output_tokens,
            output_executed,
            output_row_generations,
            ctypes.byref(model_calls),
            ctypes.byref(device_status),
            ctypes.byref(feed_calls),
            ctypes.byref(fetch_calls),
            ctypes.byref(commit_state_value),
            ctypes.byref(kv_import_checksum),
            ctypes.byref(wall_us),
            ctypes.byref(native_cpu_us),
            ctypes.byref(declared_input_bytes),
            ctypes.byref(declared_output_bytes),
            str(transfer_path).encode() if transfer_path is not None else None,
            transfer_id,
        )
        try:
            commit_state = EpochCommitState(commit_state_value.value)
        except ValueError as exc:
            raise ResidentEpochExecutionError(
                "resident_epoch_execute returned no valid commit state",
                commit_state=EpochCommitState.EXECUTING,
                status=status,
            ) from exc
        if status != 0:
            raise ResidentEpochExecutionError(
                f"resident_epoch_execute failed with status {status}",
                commit_state=commit_state,
                status=status,
            )

        tokens_by_req: dict[str, list[int]] = {}
        for request in plan.requests:
            executed = output_executed[request.row]
            if executed < 0 or executed > plan.max_steps:
                raise ResidentEpochExecutionError(
                    "native backend returned an invalid executed count",
                    commit_state=commit_state,
                )
            offset = request.row * MAX_EPOCH_STEPS
            tokens_by_req[request.req_id] = [
                int(output_tokens[offset + step]) for step in range(executed)
            ]
        return NativeEpochOutput(
            status=device_status.value,
            model_calls=model_calls.value,
            token_ids=tokens_by_req,
            row_generations=tuple(output_row_generations),
            commit_state=commit_state,
            feed_calls=feed_calls.value,
            fetch_calls=fetch_calls.value,
            wall_us=wall_us.value,
            native_cpu_us=native_cpu_us.value,
            declared_input_bytes=declared_input_bytes.value,
            declared_output_bytes=declared_output_bytes.value,
            kv_imported=transfer_path is not None,
            kv_import_checksum=kv_import_checksum.value & 0xFFFFFFFF,
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
