from typing import Any

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from .backend import ResidentEpochBackend, load_backend_from_env
from .config import ResidentEpochConfig
from .contract import get_plan


NUM_LAYERS = 28
NUM_KV_HEADS = 4
HEAD_SIZE = 128
RESIDENT_WORKER_QUALNAME = (
    "vllm_ascend_resident_epoch.worker.ResidentEpochWorker"
)


class ResidentEpochWorker(WorkerBase):
    """Dedicated worker that gives model and KV ownership to DataFlow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resident_config = ResidentEpochConfig.from_env()
        self.backend: ResidentEpochBackend | None = None
        self.kv_cache_config: KVCacheConfig | None = None
        self._owns_distributed_environment = False
        if self.parallel_config.tensor_parallel_size != 1:
            raise ValueError("resident epoch worker requires tensor parallel size 1")
        if self.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("resident epoch worker requires pipeline parallel size 1")

    def init_device(self) -> None:
        # DataFlow initializes and owns the GE/ACL device context in the native
        # backend. Creating an NPUModelRunner here would duplicate the model.
        if self.parallel_config.worker_cls == RESIDENT_WORKER_QUALNAME:
            from vllm.distributed.parallel_state import (
                ensure_model_parallel_initialized,
                init_distributed_environment,
            )

            if not torch.distributed.is_initialized():
                init_distributed_environment(
                    world_size=self.parallel_config.world_size,
                    rank=self.rank,
                    distributed_init_method=self.distributed_init_method,
                    local_rank=self.local_rank,
                    backend="gloo",
                )
                self._owns_distributed_environment = True
            ensure_model_parallel_initialized(
                self.parallel_config.tensor_parallel_size,
                self.parallel_config.pipeline_parallel_size,
                self.parallel_config.prefill_context_parallel_size,
                self.parallel_config.decode_context_parallel_size,
                backend="gloo",
            )
        self.device = torch.device("npu:0")

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        if load_dummy_weights:
            raise ValueError("resident epoch worker does not support dummy weights")
        self.backend = load_backend_from_env()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        spec = FullAttentionSpec(
            block_size=self.resident_config.block_size,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.bfloat16,
        )
        return {
            f"model.layers.{layer}.self_attn": spec for layer in range(NUM_LAYERS)
        }

    def determine_available_memory(self) -> int:
        page_size = next(iter(self.get_kv_cache_spec().values())).page_size_bytes
        return (
            self.resident_config.physical_blocks
            * NUM_LAYERS
            * page_size
        )

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        if kv_cache_config.num_blocks < self.resident_config.physical_blocks:
            raise ValueError(
                "scheduler KV capacity is smaller than the static DataFlow graph"
            )
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("resident epoch worker requires one KV cache group")
        if kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size != 128:
            raise ValueError("resident epoch worker requires 128-token KV blocks")
        self.kv_cache_config = kv_cache_config

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def execute_model(self, scheduler_output: Any):
        plan = get_plan(scheduler_output)
        if plan is None:
            if scheduler_output.total_num_scheduled_tokens == 0:
                return ModelRunnerOutput(
                    req_ids=[],
                    req_id_to_index={},
                    sampled_token_ids=[],
                )
            raise RuntimeError(
                "dedicated resident epoch worker received a request outside "
                "the native support envelope"
            )
        if self.backend is None:
            raise RuntimeError("resident epoch backend is not loaded")
        return self.backend.execute(plan)

    def sample_tokens(self, grammar_output: Any):
        raise RuntimeError("resident epoch worker returns sampled tokens synchronously")

    def get_cache_block_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.get_kv_cache_spec().values())

    def get_model(self):
        raise RuntimeError("the resident DataFlow model is not a torch.nn.Module")

    def add_lora(self, lora_request: Any) -> bool:
        raise RuntimeError("LoRA is outside the resident epoch support envelope")

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    def update_max_model_len(self, max_model_len: int) -> None:
        if max_model_len != self.model_config.max_model_len:
            raise ValueError("resident epoch worker does not support auto-fit model length")

    def shutdown(self) -> None:
        if self.backend is not None:
            close = getattr(self.backend.engine, "close", None)
            if callable(close):
                close()
            self.backend = None
        if self._owns_distributed_environment:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
            self._owns_distributed_environment = False
