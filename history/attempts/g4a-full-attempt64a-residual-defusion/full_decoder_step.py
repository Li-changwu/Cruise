#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PROMPT = "Static graph replay reduces host scheduling overhead."
NUM_LAYERS = 28
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944
NUM_HEADS = 28
NUM_KV_HEADS = 4
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
HEAD_DIM = 128
VOCAB_SIZE = 152064
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0
PHYSICAL_BLOCKS = 2
BLOCK_SIZE = 128
LOGICAL_CAPACITY = 8
RTOL = 5e-3
ATOL = 5e-3
TILING_WORDS = (28, 1, 128, 8, 16, 512, 16, 1, 1, 1, 28, 5, 2336, 24, 0, 0, 0, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_keys(index: int) -> tuple[str, ...]:
    prefix = f"model.layers.{index}"
    return (
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.q_proj.bias",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.k_proj.bias",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.v_proj.bias",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    )


def required_keys() -> tuple[str, ...]:
    keys = ["model.embed_tokens.weight"]
    for index in range(NUM_LAYERS):
        keys.extend(layer_keys(index))
    keys.extend(("model.norm.weight", "lm_head.weight"))
    return tuple(keys)


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {"model.embed_tokens.weight": (VOCAB_SIZE, HIDDEN_SIZE)}
    for index in range(NUM_LAYERS):
        prefix = f"model.layers.{index}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (HIDDEN_SIZE,),
                f"{prefix}.post_attention_layernorm.weight": (HIDDEN_SIZE,),
                f"{prefix}.self_attn.q_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
                f"{prefix}.self_attn.q_proj.bias": (HIDDEN_SIZE,),
                f"{prefix}.self_attn.k_proj.weight": (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
                f"{prefix}.self_attn.k_proj.bias": (NUM_KV_HEADS * HEAD_DIM,),
                f"{prefix}.self_attn.v_proj.weight": (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
                f"{prefix}.self_attn.v_proj.bias": (NUM_KV_HEADS * HEAD_DIM,),
                f"{prefix}.self_attn.o_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
                f"{prefix}.mlp.gate_proj.weight": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                f"{prefix}.mlp.up_proj.weight": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                f"{prefix}.mlp.down_proj.weight": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            }
        )
    shapes["model.norm.weight"] = (HIDDEN_SIZE,)
    shapes["lm_head.weight"] = (VOCAB_SIZE, HIDDEN_SIZE)
    return shapes


def audit_checkpoint(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = set(required_keys())
    observed = set(index["weight_map"])
    expected_config = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "num_hidden_layers": NUM_LAYERS,
        "vocab_size": VOCAB_SIZE,
        "rms_norm_eps": RMS_EPS,
        "rope_theta": ROPE_THETA,
    }
    config_checks = {name: config.get(name) == value for name, value in expected_config.items()}
    shards = sorted(set(index["weight_map"].values()))
    shard_sizes = {name: (model_dir / name).resolve().stat().st_size for name in shards}
    shapes = expected_shapes()
    shape_failures = {}
    dtype_failures = {}
    parameter_bytes = 0
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in index["weight_map"].items():
        keys_by_shard[shard_name].append(key)
    for shard_name in shards:
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            for key in keys_by_shard[shard_name]:
                tensor_slice = handle.get_slice(key)
                observed_shape = tuple(tensor_slice.get_shape())
                observed_dtype = str(tensor_slice.get_dtype())
                if observed_shape != shapes[key]:
                    shape_failures[key] = {"expected": shapes[key], "observed": observed_shape}
                if observed_dtype != "BF16":
                    dtype_failures[key] = observed_dtype
                parameter_bytes += math.prod(observed_shape) * 2
    valid = (
        model_dir.name == MODEL_REVISION
        and expected == observed
        and all(config_checks.values())
        and all((model_dir / name).is_file() for name in shards)
        and not shape_failures
        and not dtype_failures
    )
    return {
        "valid": valid,
        "revision": model_dir.name,
        "tensor_count": len(observed),
        "required_tensor_count": len(expected),
        "missing_keys": sorted(expected - observed),
        "unexpected_keys": sorted(observed - expected),
        "config_checks": config_checks,
        "shape_failures": shape_failures,
        "dtype_failures": dtype_failures,
        "parameter_bytes": parameter_bytes,
        "index_sha256": sha256(index_path),
        "shard_sizes": shard_sizes,
        "checkpoint_bytes": sum(shard_sizes.values()),
    }


def load_checkpoint(model_dir: Path) -> dict[str, torch.Tensor]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in required_keys():
        grouped[index["weight_map"][key]].append(key)
    result: dict[str, torch.Tensor] = {}
    for shard_name in sorted(grouped):
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            for key in grouped[shard_name]:
                tensor = handle.get_tensor(key).contiguous()
                if tensor.dtype != torch.bfloat16:
                    raise RuntimeError(f"unexpected checkpoint dtype {key}: {tensor.dtype}")
                result[key] = tensor
    if set(result) != set(required_keys()):
        raise RuntimeError("checkpoint load did not produce exactly the required 339 tensors")
    return result


class LayerWeights(torch.nn.Module):
    def __init__(self, values: dict[str, torch.Tensor], index: int) -> None:
        super().__init__()
        prefix = f"model.layers.{index}"
        names = {
            "input_norm": f"{prefix}.input_layernorm.weight",
            "post_norm": f"{prefix}.post_attention_layernorm.weight",
            "q_weight": f"{prefix}.self_attn.q_proj.weight",
            "q_bias": f"{prefix}.self_attn.q_proj.bias",
            "k_weight": f"{prefix}.self_attn.k_proj.weight",
            "k_bias": f"{prefix}.self_attn.k_proj.bias",
            "v_weight": f"{prefix}.self_attn.v_proj.weight",
            "v_bias": f"{prefix}.self_attn.v_proj.bias",
            "o_weight": f"{prefix}.self_attn.o_proj.weight",
            "gate_weight": f"{prefix}.mlp.gate_proj.weight",
            "up_weight": f"{prefix}.mlp.up_proj.weight",
            "down_weight": f"{prefix}.mlp.down_proj.weight",
        }
        for local_name, checkpoint_name in names.items():
            self.register_buffer(local_name, values.pop(checkpoint_name))


class PagedQwenDecoderStep(torch.nn.Module):
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer("embedding", values.pop("model.embed_tokens.weight"))
        self.layers = torch.nn.ModuleList([LayerWeights(values, index) for index in range(NUM_LAYERS)])
        self.register_buffer("final_norm", values.pop("model.norm.weight"))
        self.register_buffer("lm_head", values.pop("lm_head.weight"))
        if values:
            raise RuntimeError(f"unconsumed checkpoint tensors: {sorted(values)}")
        inv_freq = 1.0 / (
            ROPE_THETA
            ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
        )
        self.register_buffer("inv_freq", inv_freq.contiguous())
        self.register_buffer(
            "physical_slots",
            torch.arange(PHYSICAL_BLOCKS * BLOCK_SIZE, dtype=torch.int32),
        )
        self.register_buffer("logical_offsets", torch.arange(LOGICAL_CAPACITY, dtype=torch.int32))

    @staticmethod
    def rms_norm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        hidden_fp32 = hidden.float()
        variance = hidden_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = (hidden_fp32 * torch.rsqrt(variance + RMS_EPS)).to(hidden.dtype)
        return normalized * weight

    @staticmethod
    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        return torch.cat((-value[..., HEAD_DIM // 2 :], value[..., : HEAD_DIM // 2]), dim=-1)

    def rope(self, position: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.matmul(
            self.inv_freq.view(1, HEAD_DIM // 2, 1),
            position.float().view(1, 1, 1),
        ).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)

    def update_and_read_cache(
        self,
        layer_cache: torch.Tensor,
        new_value: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = layer_cache.reshape(PHYSICAL_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        slot_mask = (self.physical_slots == slot_mapping.reshape(1)).view(-1, 1, 1)
        replacement = new_value.reshape(1, NUM_KV_HEADS, HEAD_DIM)
        updated_flat = torch.where(slot_mask, replacement, flat)
        logical_blocks = self.logical_offsets // BLOCK_SIZE
        offsets = self.logical_offsets - logical_blocks * BLOCK_SIZE
        physical_blocks = torch.index_select(
            block_table.reshape(-1), 0, logical_blocks.to(torch.int64)
        )
        dense_slots = (physical_blocks * BLOCK_SIZE + offsets).to(torch.int64)
        dense = torch.index_select(updated_flat, 0, dense_slots)
        return updated_flat.reshape_as(layer_cache), dense.permute(1, 0, 2).contiguous()

    def forward(
        self,
        token_id: torch.Tensor,
        position: torch.Tensor,
        sequence_length: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        explicit_tiling: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.embedding(token_id, self.embedding).view(1, 1, HIDDEN_SIZE)
        cos, sin = self.rope(position, hidden.dtype)
        cos = cos.view(1, 1, 1, HEAD_DIM)
        sin = sin.view(1, 1, 1, HEAD_DIM)
        valid = self.logical_offsets < sequence_length.reshape(1)
        new_key_layers = []
        new_value_layers = []
        layer_hiddens = []

        for layer_index, layer in enumerate(self.layers):
            residual = hidden
            normalized = self.rms_norm(hidden, layer.input_norm)
            normalized_2d = normalized.reshape(-1, HIDDEN_SIZE)
            query = torch.ops.g4a_linear_t_bias.linear(
                normalized_2d, layer.q_weight, layer.q_bias
            ).reshape(1, 1, HIDDEN_SIZE)
            key = torch.ops.g4a_linear_t_bias.linear(
                normalized_2d, layer.k_weight, layer.k_bias
            ).reshape(1, 1, NUM_KV_HEADS * HEAD_DIM)
            value = torch.ops.g4a_linear_t_bias.linear(
                normalized_2d, layer.v_weight, layer.v_bias
            ).reshape(1, 1, NUM_KV_HEADS * HEAD_DIM)
            query = query.view(1, 1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            key = key.view(1, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            value = value.view(1, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            query = query * cos + self.rotate_half(query) * sin
            key = key * cos + self.rotate_half(key) * sin

            updated_key, dense_key = self.update_and_read_cache(
                key_cache[layer_index], key, block_table, slot_mapping
            )
            updated_value, dense_value = self.update_and_read_cache(
                value_cache[layer_index], value, block_table, slot_mapping
            )
            new_key_layers.append(updated_key)
            new_value_layers.append(updated_value)
            expanded_key = (
                dense_key.unsqueeze(1)
                .expand(NUM_KV_HEADS, NUM_KV_GROUPS, LOGICAL_CAPACITY, HEAD_DIM)
                .reshape(NUM_HEADS, LOGICAL_CAPACITY, HEAD_DIM)
            )
            expanded_value = (
                dense_value.unsqueeze(1)
                .expand(NUM_KV_HEADS, NUM_KV_GROUPS, LOGICAL_CAPACITY, HEAD_DIM)
                .reshape(1, NUM_HEADS, LOGICAL_CAPACITY, HEAD_DIM)
            )
            raw_qk = torch.ops.g4a_qk.exact_qk(
                query.squeeze(2).contiguous(),
                expanded_key.transpose(1, 2).contiguous(),
                explicit_tiling,
            )
            scaled_scores = (raw_qk.float() / math.sqrt(HEAD_DIM)).to(torch.bfloat16).unsqueeze(2)
            scores = torch.ops.g4a_barrier.materialize(scaled_scores)
            scores = torch.where(
                valid.view(1, 1, 1, LOGICAL_CAPACITY),
                scores,
                torch.full_like(scores, torch.finfo(torch.bfloat16).min),
            )
            probabilities = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
            attention = torch.matmul(probabilities, expanded_value)
            attention = attention.transpose(1, 2).reshape(1, 1, HIDDEN_SIZE)
            attention_projection = torch.ops.g4a_linear_t.mm_t(
                attention.reshape(-1, HIDDEN_SIZE), layer.o_weight
            ).reshape(1, 1, HIDDEN_SIZE)
            attention_projection = torch.ops.g4a_materialize.materialize(
                attention_projection
            )
            hidden = residual + attention_projection

            residual = hidden
            normalized = self.rms_norm(hidden, layer.post_norm)
            normalized_2d = normalized.reshape(-1, HIDDEN_SIZE)
            gate_preactivation = torch.ops.g4a_linear_t.mm_t(
                normalized_2d, layer.gate_weight
            ).reshape(1, 1, INTERMEDIATE_SIZE)
            gate = F.silu(gate_preactivation)
            up = torch.ops.g4a_linear_t.mm_t(
                normalized_2d, layer.up_weight
            ).reshape(1, 1, INTERMEDIATE_SIZE)
            mlp_product = gate * up
            mlp_projection = torch.ops.g4a_linear_t.mm_t(
                mlp_product.reshape(-1, INTERMEDIATE_SIZE), layer.down_weight
            ).reshape(1, 1, HIDDEN_SIZE)
            mlp_projection = torch.ops.g4a_materialize.materialize(mlp_projection)
            hidden = residual + mlp_projection
            layer_hiddens.append(hidden)

        hidden = self.rms_norm(hidden, self.final_norm)
        logits = torch.ops.g4a_linear_t.mm_t(
            hidden.reshape(-1, HIDDEN_SIZE), self.lm_head
        ).reshape(1, 1, VOCAB_SIZE).float()
        return (
            logits,
            torch.stack(new_key_layers),
            torch.stack(new_value_layers),
            position + 1,
            torch.stack(layer_hiddens),
        )


def bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().contiguous().view(torch.uint16).cpu().numpy().copy()


def bits_to_bf16(value: np.ndarray) -> torch.Tensor:
    bits = torch.from_numpy(np.ascontiguousarray(value.astype(np.uint16, copy=False)))
    return bits.view(torch.bfloat16)


def tensor_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual_fp32 = actual.astype(np.float32)
    expected_fp32 = expected.astype(np.float32)
    close = np.isclose(actual_fp32, expected_fp32, rtol=RTOL, atol=ATOL)
    exact = np.equal(actual, expected)
    return {
        "all_within_tolerance": bool(np.all(close)),
        "all_exact": bool(np.all(exact)),
        "max_abs_error": float(np.max(np.abs(actual_fp32 - expected_fp32))),
        "tolerance_failure_count": int(close.size - np.count_nonzero(close)),
        "exact_mismatch_count": int(exact.size - np.count_nonzero(exact)),
    }


def cache_layer(cache, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        layer = cache.layers[index]
        keys = getattr(layer, "keys", getattr(layer, "key_cache", None))
        values = getattr(layer, "values", getattr(layer, "value_cache", None))
        if keys is not None and values is not None:
            return keys, values
    return cache[index]


def initial_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    generator = torch.Generator(device="cpu").manual_seed(530041)
    key = (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.125).to(torch.bfloat16)
    value = (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.125).to(torch.bfloat16)
    block_table = torch.tensor([[1, 0]], dtype=torch.int32)
    tiling = torch.from_numpy(np.asarray(TILING_WORDS, dtype="<u4").view(np.uint8).copy())
    return key, value, block_table, tiling


def register_custom_ops(
    exact_qk_source: Path, barrier_source: Path, materialize_source: Path
) -> None:
    sys.path.insert(0, str(exact_qk_source))
    sys.path.insert(0, str(barrier_source))
    sys.path.insert(0, str(materialize_source))
    import torch_npu  # noqa: F401
    import export_qk_attempt50_air  # noqa: F401
    import export_barrier_probe  # noqa: F401
    import export_probe  # noqa: F401
    import g4a_linear_t  # noqa: F401
    import g4a_linear_t_bias  # noqa: F401


def official_eager_reference(model_dir: Path, token_ids: list[int]) -> list[dict]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().npu()
    cache = None
    steps = []
    with torch.no_grad():
        for step, token_id in enumerate(token_ids):
            token = torch.tensor([[token_id]], dtype=torch.int64, device="npu")
            position = torch.tensor([[step]], dtype=torch.int64, device="npu")
            outputs = model(
                input_ids=token,
                position_ids=position,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = outputs.past_key_values
            written_key, written_value = [], []
            for layer_index in range(NUM_LAYERS):
                key, value = cache_layer(cache, layer_index)
                written_key.append(bf16_bits(key[:, :, step : step + 1, :]))
                written_value.append(bf16_bits(value[:, :, step : step + 1, :]))
            logits = outputs.logits.detach().float().cpu().numpy().copy()
            steps.append(
                {
                    "logits": logits,
                    "greedy": int(np.argmax(logits.reshape(-1))),
                    "written_key_bits": np.concatenate(written_key, axis=0),
                    "written_value_bits": np.concatenate(written_value, axis=0),
                }
            )
    del cache, model
    gc.collect()
    torch.npu.empty_cache()
    return steps


def eager_screen(
    model_dir: Path,
    exact_qk_source: Path,
    barrier_source: Path,
    materialize_source: Path,
    output_dir: Path,
) -> dict:
    from transformers import AutoTokenizer

    register_custom_ops(exact_qk_source, barrier_source, materialize_source)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    token_ids = tokenizer.encode(PROMPT, add_special_tokens=False)[:4]
    if len(token_ids) != 4:
        raise RuntimeError(f"expected four prompt tokens, got {token_ids}")
    official = official_eager_reference(model_dir, token_ids)
    values = load_checkpoint(model_dir)
    model = PagedQwenDecoderStep(values).eval().npu()
    key_cpu, value_cpu, block_table_cpu, tiling_cpu = initial_state()
    initial_key_bits = bf16_bits(key_cpu)
    initial_value_bits = bf16_bits(value_cpu)
    key = key_cpu.npu()
    value = value_cpu.npu()
    block_table = block_table_cpu.npu()
    tiling = tiling_cpu.npu()
    reference: dict[str, np.ndarray] = {
        "token_ids": np.asarray(token_ids, dtype=np.int64),
        "input_key_cache_bits": initial_key_bits,
        "input_value_cache_bits": initial_value_bits,
        "block_table": block_table_cpu.numpy(),
        "tiling": tiling_cpu.numpy(),
    }
    step_results = []
    with torch.no_grad():
        for step, token_id in enumerate(token_ids, start=1):
            token = torch.tensor([[token_id]], dtype=torch.int64, device="npu")
            position = torch.tensor([step - 1], dtype=torch.int64, device="npu")
            sequence_length = torch.tensor([[step]], dtype=torch.int32, device="npu")
            slot_mapping = torch.tensor([BLOCK_SIZE + step - 1], dtype=torch.int32, device="npu")
            logits, key, value, next_position, layer_hiddens = model(
                token, position, sequence_length, block_table, slot_mapping, key, value, tiling
            )
            torch.npu.synchronize()
            logits_np = logits.detach().cpu().numpy().copy()
            key_bits = bf16_bits(key)
            value_bits = bf16_bits(value)
            layer_hidden_bits = bf16_bits(layer_hiddens)
            written_key = key_bits[:, 1, step - 1 : step, :, :].transpose(0, 2, 1, 3)
            written_value = value_bits[:, 1, step - 1 : step, :, :].transpose(0, 2, 1, 3)
            logits_metric = tensor_metrics(logits_np, official[step - 1]["logits"])
            key_metric = tensor_metrics(
                (written_key.astype(np.uint32) << 16).view(np.float32),
                (official[step - 1]["written_key_bits"].astype(np.uint32) << 16).view(np.float32),
            )
            value_metric = tensor_metrics(
                (written_value.astype(np.uint32) << 16).view(np.float32),
                (official[step - 1]["written_value_bits"].astype(np.uint32) << 16).view(np.float32),
            )
            addressed = BLOCK_SIZE + step - 1
            flat_key = key_bits.reshape(NUM_LAYERS, PHYSICAL_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
            flat_value = value_bits.reshape(NUM_LAYERS, PHYSICAL_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
            previous_key = initial_key_bits.reshape(flat_key.shape) if step == 1 else previous_key
            previous_value = initial_value_bits.reshape(flat_value.shape) if step == 1 else previous_value
            mask = np.ones(flat_key.shape[1], dtype=bool)
            mask[addressed] = False
            unwritten_key_exact = bool(np.array_equal(flat_key[:, mask], previous_key[:, mask]))
            unwritten_value_exact = bool(np.array_equal(flat_value[:, mask], previous_value[:, mask]))
            greedy = int(np.argmax(logits_np.reshape(-1)))
            item = {
                "step": step,
                "position": step - 1,
                "sequence_length": step,
                "slot_mapping": addressed,
                "logits_vs_hf": logits_metric,
                "written_key_vs_hf": key_metric,
                "written_value_vs_hf": value_metric,
                "manual_greedy": greedy,
                "hf_greedy": official[step - 1]["greedy"],
                "greedy_equal": greedy == official[step - 1]["greedy"],
                "unwritten_key_elementwise_exact": unwritten_key_exact,
                "unwritten_value_elementwise_exact": unwritten_value_exact,
                "next_position": int(next_position.item()),
            }
            item["pass"] = (
                logits_metric["all_within_tolerance"]
                and key_metric["all_within_tolerance"]
                and value_metric["all_within_tolerance"]
                and item["greedy_equal"]
                and unwritten_key_exact
                and unwritten_value_exact
                and item["next_position"] == step
            )
            step_results.append(item)
            reference[f"step{step}_logits"] = logits_np
            reference[f"step{step}_key_cache_bits"] = key_bits
            reference[f"step{step}_value_cache_bits"] = value_bits
            reference[f"step{step}_next_position"] = np.asarray([item["next_position"]], dtype=np.int64)
            reference[f"step{step}_layer_hidden_bits"] = layer_hidden_bits
            previous_key = flat_key.copy()
            previous_value = flat_value.copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "attempt64a-eager-reference.npz"
    np.savez(reference_path, **reference)
    result = {
        "gate": "G4a Attempt 64a residual-defusion eager screen",
        "pass": all(item["pass"] for item in step_results),
        "model_revision": model_dir.name,
        "token_ids": token_ids,
        "tokens": tokenizer.convert_ids_to_tokens(token_ids),
        "paged_kv_shape_per_tensor": [NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM],
        "block_table": block_table_cpu.tolist(),
        "logical_capacity": LOGICAL_CAPACITY,
        "rtol": RTOL,
        "atol": ATOL,
        "reference_sha256": sha256(reference_path),
        "steps": step_results,
        "claim_boundary": "Diagnostic output only; this experiment cannot pass G4a or enter G4b.",
    }
    result_path = output_dir / "eager-screen.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("G4A_ATTEMPT64A_EAGER " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)
    return result


def export_air(
    model_dir: Path,
    exact_qk_source: Path,
    barrier_source: Path,
    materialize_source: Path,
    reference_path: Path,
    output_dir: Path,
) -> dict:
    register_custom_ops(exact_qk_source, barrier_source, materialize_source)
    import torch_npu  # noqa: F401
    from torchair.configs.compiler_config import CompilerConfig
    from torchair.npu_export import dynamo_export

    reference = np.load(reference_path)
    values = load_checkpoint(model_dir)
    model = PagedQwenDecoderStep(values).eval().npu()
    token = torch.from_numpy(reference["token_ids"][:1].copy()).reshape(1, 1).npu()
    position = torch.zeros((1,), dtype=torch.int64, device="npu")
    sequence_length = torch.ones((1, 1), dtype=torch.int32, device="npu")
    block_table = torch.from_numpy(reference["block_table"].copy()).npu()
    slot_mapping = torch.tensor([BLOCK_SIZE], dtype=torch.int32, device="npu")
    key = bits_to_bf16(reference["input_key_cache_bits"]).npu()
    value = bits_to_bf16(reference["input_value_cache_bits"]).npu()
    tiling = torch.from_numpy(reference["tiling"].copy()).npu()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        token,
        position,
        sequence_length,
        block_table,
        slot_mapping,
        key,
        value,
        tiling,
        model=model,
        export_path=str(output_dir),
        export_name="qwen_full_decoder_step_attempt64a",
        dynamic=False,
        config=config,
    )
    air = output_dir / "qwen_full_decoder_step_attempt64a.air"
    graph = output_dir / "dynamo.pbtxt"
    external_files = [path for path in output_dir.iterdir() if path.is_file() and path.suffix == ""]
    result = {
        "gate": "G4a Attempt 64a residual-defusion AIR export",
        "execution_success": air.is_file() and graph.is_file(),
        "air_sha256": sha256(air) if air.is_file() else None,
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "external_file_count": len(external_files),
        "external_file_bytes": sum(path.stat().st_size for path in external_files),
        "reference_sha256": sha256(reference_path),
        "claim_boundary": "Diagnostic AIR only; it cannot pass G4a.",
    }
    (output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("G4A_ATTEMPT64A_EXPORT " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["execution_success"]:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "eager", "export"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--exact-qk-source", type=Path)
    parser.add_argument("--barrier-source", type=Path)
    parser.add_argument("--materialize-source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_checkpoint(args.model_dir)
    (args.output_dir / "checkpoint-audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("G4A_ATTEMPT53_AUDIT " + json.dumps(audit, ensure_ascii=True), flush=True)
    if not audit["valid"]:
        raise SystemExit(2)
    if args.mode == "audit":
        return
    if (
        args.exact_qk_source is None
        or args.barrier_source is None
        or args.materialize_source is None
    ):
        parser.error(
            "--exact-qk-source, --barrier-source and --materialize-source "
            "are required for eager/export"
        )
    torch.npu.set_device(0)
    if args.mode == "eager":
        eager_screen(
            args.model_dir,
            args.exact_qk_source,
            args.barrier_source,
            args.materialize_source,
            args.output_dir,
        )
    else:
        if args.reference is None:
            parser.error("--reference is required for export")
        export_air(
            args.model_dir,
            args.exact_qk_source,
            args.barrier_source,
            args.materialize_source,
            args.reference,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
