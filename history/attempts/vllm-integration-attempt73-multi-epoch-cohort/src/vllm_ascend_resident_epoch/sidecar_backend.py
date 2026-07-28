import os
import socket
import struct
import subprocess
import time
from pathlib import Path

from .backend import NativeEpochOutput, NativeWarmupOutput
from .contract import ResidentEpochPlan


REQUEST_MAGIC = 0x71317131
RESPONSE_MAGIC = 0x71327132
PROTOCOL_VERSION = 2
EXECUTE = 1
WARM_UP = 2
SHUTDOWN = 3
GRAPH_BATCH_SIZE = 4
MAX_EPOCH_STEPS = 8
WARMUP_GENERATION = 2**31 - 1
REQUEST = struct.Struct("<IHHii4q4q4i4i4i")
RESPONSE = struct.Struct("<Iiiiiiq4i4i32q")


def _required_path(name: str, *, directory: bool = False) -> Path:
    raw = os.getenv(name)
    if not raw:
        raise RuntimeError(f"{name} is required by the DataFlow sidecar")
    path = Path(raw).resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise RuntimeError(f"{name} must name an existing {kind}: {path}")
    return path


def _by_row(plan: ResidentEpochPlan, field: str, fill: int) -> list[int]:
    values = [fill] * GRAPH_BATCH_SIZE
    for request in plan.requests:
        values[request.row] = int(getattr(request, field))
    return values


class SidecarDataFlowEngine:
    def __init__(self) -> None:
        server = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_SERVER")
        air = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_AIR")
        graph_config = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_GRAPH_CONFIG")
        func_config = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_FUNC_CONFIG")
        tiling = _required_path("VLLM_ASCEND_RESIDENT_EPOCH_TILING")
        weights = _required_path(
            "VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS", directory=True
        )
        socket_raw = os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_SOCKET")
        if not socket_raw:
            raise RuntimeError("VLLM_ASCEND_RESIDENT_EPOCH_SOCKET is required")
        self.socket_path = Path(socket_raw).resolve()
        if not str(self.socket_path).startswith("/dev/shm/"):
            raise RuntimeError("resident epoch socket must be under /dev/shm")
        if not str(weights).startswith("/dev/shm/"):
            raise RuntimeError("resident epoch external weights must be in /dev/shm")

        try:
            self.socket_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"cannot reset sidecar socket: {exc}") from exc

        self.process = subprocess.Popen(
            [
                str(server),
                str(self.socket_path),
                str(air),
                str(graph_config),
                str(func_config),
                str(weights),
                str(tiling),
            ],
            close_fds=True,
        )
        self.socket: socket.socket | None = None
        startup_timeout = int(
            os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_STARTUP_TIMEOUT", "3600")
        )
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "resident epoch sidecar exited during startup with status "
                    f"{self.process.returncode}"
                )
            if self.socket_path.exists():
                candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    candidate.connect(str(self.socket_path))
                except OSError:
                    candidate.close()
                else:
                    self.socket = candidate
                    break
            time.sleep(0.05)
        if self.socket is None:
            self._stop_process()
            raise TimeoutError("resident epoch sidecar socket startup timed out")
        self.socket.settimeout(startup_timeout)
        startup = self._receive_response()
        if startup[1] != 0:
            status = startup[1]
            self.close(force=True)
            raise RuntimeError(
                f"resident epoch sidecar native initialization failed: {status}"
            )

    def _receive_exact(self, size: int) -> bytes:
        assert self.socket is not None
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self.socket.recv(remaining)
            if not chunk:
                return_code = self.process.poll()
                raise RuntimeError(
                    "resident epoch sidecar closed the control socket"
                    f" (status={return_code})"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _receive_response(self) -> tuple[int, ...]:
        values = RESPONSE.unpack(self._receive_exact(RESPONSE.size))
        if values[0] != RESPONSE_MAGIC:
            raise RuntimeError("resident epoch sidecar response magic mismatch")
        return values

    def warm_up(self) -> NativeWarmupOutput:
        if self.socket is None:
            raise RuntimeError("resident epoch sidecar is closed")
        payload = REQUEST.pack(
            REQUEST_MAGIC,
            PROTOCOL_VERSION,
            WARM_UP,
            1,
            1,
            11690,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            151645,
            151645,
            151645,
            151645,
            WARMUP_GENERATION,
            0,
            0,
            0,
        )
        self.socket.sendall(payload)
        values = self._receive_response()
        if values[1] != 0:
            raise RuntimeError(f"resident epoch warmup failed: {values[1]}")
        if values[7:11] != (1, 0, 0, 0):
            raise RuntimeError("resident epoch warmup did not execute exactly one row")
        if values[11:15] != (WARMUP_GENERATION, 0, 0, 0):
            raise RuntimeError("resident epoch warmup generation acknowledgement failed")
        return NativeWarmupOutput(
            status=values[2],
            model_calls=values[3],
            feed_calls=values[4],
            fetch_calls=values[5],
            wall_us=values[6],
        )

    def execute(self, plan: ResidentEpochPlan) -> NativeEpochOutput:
        if self.socket is None:
            raise RuntimeError("resident epoch sidecar is closed")
        if plan.graph_batch_size != GRAPH_BATCH_SIZE:
            raise ValueError("the sidecar backend owns only the static B=4 graph")
        requests = list(plan.requests)
        payload = REQUEST.pack(
            REQUEST_MAGIC,
            PROTOCOL_VERSION,
            EXECUTE,
            len(requests),
            plan.max_steps,
            *_by_row(plan, "token_id", 0),
            *_by_row(plan, "position", 0),
            *_by_row(plan, "sequence_length", 0),
            *_by_row(plan, "eos_token_id", 151645),
            *plan.row_generations,
        )
        self.socket.sendall(payload)
        values = self._receive_response()
        transport_status = values[1]
        if transport_status != 0:
            raise RuntimeError(
                f"resident epoch sidecar execute failed: {transport_status}"
            )
        device_status, model_calls, feed_calls, fetch_calls, wall_us = values[2:7]
        executed = values[7:11]
        row_generations = tuple(values[11:15])
        flat_tokens = values[15:]
        token_ids: dict[str, list[int]] = {}
        for request in requests:
            count = executed[request.row]
            if count < 0 or count > plan.max_steps:
                raise RuntimeError("sidecar returned an invalid executed count")
            offset = request.row * MAX_EPOCH_STEPS
            token_ids[request.req_id] = list(flat_tokens[offset : offset + count])
        return NativeEpochOutput(
            status=device_status,
            model_calls=model_calls,
            token_ids=token_ids,
            row_generations=row_generations,
            feed_calls=feed_calls,
            fetch_calls=fetch_calls,
            wall_us=wall_us,
        )

    def _stop_process(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=60)

    def close(self, *, force: bool = False) -> None:
        if self.socket is not None:
            if not force and self.process.poll() is None:
                try:
                    self.socket.sendall(
                        REQUEST.pack(
                            REQUEST_MAGIC,
                            PROTOCOL_VERSION,
                            SHUTDOWN,
                            0,
                            0,
                            *([0] * 20),
                        )
                    )
                    self._receive_response()
                except (OSError, RuntimeError):
                    force = True
            self.socket.close()
            self.socket = None
        if force:
            self._stop_process()
        else:
            try:
                self.process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                self._stop_process()
        try:
            self.socket_path.unlink(missing_ok=True)
        except OSError:
            pass

    def __del__(self) -> None:
        try:
            self.close(force=True)
        except Exception:
            pass


def create_sidecar_engine() -> SidecarDataFlowEngine:
    return SidecarDataFlowEngine()
