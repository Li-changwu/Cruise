from dataclasses import dataclass
from importlib import import_module
import os
from typing import Protocol

from vllm.v1.outputs import ModelRunnerOutput

from .contract import (
    CONTRACT_VERSION,
    EpochCommitState,
    ResidentEpochPlan,
    ResidentEpochResult,
    ResidentEpochExecutionError,
    attach_result,
)
from .kv_transfer import ResidentKVSnapshot


@dataclass(frozen=True)
class NativeEpochOutput:
    status: int
    model_calls: int
    token_ids: dict[str, list[int]]
    row_generations: tuple[int, ...]
    commit_state: EpochCommitState = EpochCommitState.COMMITTED
    feed_calls: int = 1
    fetch_calls: int = 1
    wall_us: int = 0
    native_cpu_us: int = 0
    declared_input_bytes: int = 0
    declared_output_bytes: int = 0
    socket_send_calls: int = 1
    socket_receive_calls: int = 1
    kv_imported: bool = False
    kv_import_checksum: int = 0


@dataclass(frozen=True)
class NativeWarmupOutput:
    status: int
    model_calls: int
    feed_calls: int
    fetch_calls: int
    wall_us: int
    commit_state: EpochCommitState = EpochCommitState.COMMITTED
    native_cpu_us: int = 0
    declared_input_bytes: int = 0
    declared_output_bytes: int = 0
    socket_send_calls: int = 1
    socket_receive_calls: int = 1


class NativeEpochEngine(Protocol):
    def warm_up(self) -> NativeWarmupOutput: ...

    def execute(self, plan: ResidentEpochPlan) -> NativeEpochOutput: ...

    def execute_with_import(
        self, plan: ResidentEpochPlan, snapshot: ResidentKVSnapshot
    ) -> NativeEpochOutput: ...


class ResidentEpochBackend:
    def __init__(self, engine: NativeEpochEngine):
        self.engine = engine

    def warm_up(self) -> NativeWarmupOutput:
        output = self.engine.warm_up()
        if output.status != 0:
            raise RuntimeError(f"native resident warmup returned {output.status}")
        if (
            output.model_calls != 1
            or output.feed_calls != 1
            or output.fetch_calls != 1
            or output.commit_state != EpochCommitState.COMMITTED
        ):
            raise RuntimeError("resident warmup must execute one complete decoder step")
        return output

    def close(self) -> None:
        close = getattr(self.engine, "close", None)
        if callable(close):
            close()

    def execute(
        self,
        plan: ResidentEpochPlan,
        snapshot: ResidentKVSnapshot | None = None,
    ) -> ModelRunnerOutput:
        try:
            plan.validate()
        except Exception as exc:
            raise ResidentEpochExecutionError(
                f"resident epoch plan validation failed: {exc}",
                commit_state=EpochCommitState.PREPARED,
            ) from exc
        try:
            if snapshot is None:
                native_output = self.engine.execute(plan)
            else:
                execute_with_import = getattr(self.engine, "execute_with_import", None)
                if not callable(execute_with_import):
                    raise ResidentEpochExecutionError(
                        "resident backend does not support KV import",
                        commit_state=EpochCommitState.PREPARED,
                    )
                native_output = execute_with_import(plan, snapshot)
        except ResidentEpochExecutionError:
            raise
        except Exception as exc:
            raise ResidentEpochExecutionError(
                "native resident epoch failed without commit-state proof",
                commit_state=EpochCommitState.EXECUTING,
            ) from exc
        if native_output.status != 0:
            raise ResidentEpochExecutionError(
                "native resident epoch returned status "
                f"{native_output.status}",
                commit_state=native_output.commit_state,
                status=native_output.status,
            )
        if native_output.commit_state != EpochCommitState.COMMITTED:
            raise ResidentEpochExecutionError(
                "successful native resident epoch was not committed",
                commit_state=native_output.commit_state,
            )
        if (
            snapshot is not None
            and native_output.kv_import_checksum != snapshot.checksum
        ):
            raise ResidentEpochExecutionError(
                "resident KV import checksum disagrees with the Host snapshot",
                commit_state=EpochCommitState.COMMITTED,
            )
        if set(native_output.token_ids) != set(plan.req_ids):
            raise ResidentEpochExecutionError(
                "native resident epoch returned the wrong request set",
                commit_state=EpochCommitState.COMMITTED,
            )

        req_ids = list(plan.req_ids)
        sampled = [native_output.token_ids[req_id] for req_id in req_ids]
        result = ResidentEpochResult(
            version=CONTRACT_VERSION,
            route="device",
            status=0,
            model_calls=native_output.model_calls,
            computed_steps={
                req_id: len(native_output.token_ids[req_id]) for req_id in req_ids
            },
            row_generations=native_output.row_generations,
            commit_state=native_output.commit_state,
            feed_calls=native_output.feed_calls,
            fetch_calls=native_output.fetch_calls,
            wall_us=native_output.wall_us,
            native_cpu_us=native_output.native_cpu_us,
            declared_input_bytes=native_output.declared_input_bytes,
            declared_output_bytes=native_output.declared_output_bytes,
            socket_send_calls=native_output.socket_send_calls,
            socket_receive_calls=native_output.socket_receive_calls,
            kv_imported=native_output.kv_imported,
            kv_import_checksum=native_output.kv_import_checksum,
            kv_snapshot_checksum=snapshot.checksum if snapshot is not None else 0,
        )
        try:
            result.validate_against(
                plan,
                {req_id: native_output.token_ids[req_id] for req_id in req_ids},
            )
        except Exception as exc:
            raise ResidentEpochExecutionError(
                f"committed resident epoch result is invalid: {exc}",
                commit_state=EpochCommitState.COMMITTED,
            ) from exc
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
        )
        attach_result(output, result)
        return output


def load_backend_from_env() -> ResidentEpochBackend:
    target = os.getenv("VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY")
    if not target or ":" not in target:
        raise RuntimeError(
            "VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY must name "
            "a module:factory for the native DataFlow backend"
        )
    module_name, factory_name = target.split(":", 1)
    factory = getattr(import_module(module_name), factory_name)
    engine = factory()
    return ResidentEpochBackend(engine)
