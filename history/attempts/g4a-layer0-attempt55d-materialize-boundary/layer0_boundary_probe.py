#!/usr/bin/env python3
import argparse
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


HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944
NUM_HEADS = 28
NUM_KV_HEADS = 4
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
HEAD_DIM = 128
PHYSICAL_BLOCKS = 2
BLOCK_SIZE = 128
LOGICAL_CAPACITY = 8
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0
TILING_WORDS = (28, 1, 128, 8, 16, 512, 16, 1, 1, 1, 28, 5, 2336, 24, 0, 0, 0, 0)

WEIGHT_KEYS = {
    "input_norm": "model.layers.0.input_layernorm.weight",
    "post_norm": "model.layers.0.post_attention_layernorm.weight",
    "q_weight": "model.layers.0.self_attn.q_proj.weight",
    "q_bias": "model.layers.0.self_attn.q_proj.bias",
    "k_weight": "model.layers.0.self_attn.k_proj.weight",
    "k_bias": "model.layers.0.self_attn.k_proj.bias",
    "v_weight": "model.layers.0.self_attn.v_proj.weight",
    "v_bias": "model.layers.0.self_attn.v_proj.bias",
    "o_weight": "model.layers.0.self_attn.o_proj.weight",
    "gate_weight": "model.layers.0.mlp.gate_proj.weight",
    "up_weight": "model.layers.0.mlp.up_proj.weight",
    "down_weight": "model.layers.0.mlp.down_proj.weight",
}

OUTPUT_NAMES = (
    "updated_key",
    "updated_value",
    "next_position",
    "input_norm",
    "query_rope",
    "key_rope",
    "value_projection",
    "masked_scores",
    "probabilities",
    "attention_value",
    "attention_projection",
    "hidden_after_attention",
    "post_attention_norm",
    "gate",
    "up",
    "mlp_product",
    "mlp_projection",
    "hidden_after_mlp",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().contiguous().view(torch.uint16).cpu().numpy()


def bits_to_bf16(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.uint16)).view(torch.bfloat16)


def load_layer0(model_dir: Path, token_ids: np.ndarray) -> tuple[dict, torch.Tensor]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    requested = dict(WEIGHT_KEYS)
    requested["embedding"] = "model.embed_tokens.weight"
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for local_name, checkpoint_name in requested.items():
        grouped[index["weight_map"][checkpoint_name]].append((local_name, checkpoint_name))

    weights = {}
    hidden_rows = None
    for shard_name, names in grouped.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            for local_name, checkpoint_name in names:
                if local_name == "embedding":
                    embedding = handle.get_slice(checkpoint_name)
                    hidden_rows = torch.cat(
                        [embedding[int(token) : int(token) + 1] for token in token_ids], dim=0
                    ).reshape(len(token_ids), 1, HIDDEN_SIZE).contiguous()
                else:
                    weights[local_name] = handle.get_tensor(checkpoint_name).contiguous()
    if set(weights) != set(WEIGHT_KEYS) or hidden_rows is None:
        raise RuntimeError("failed to load the exact layer-0 weight and embedding subset")
    return weights, hidden_rows


class Layer0BoundaryProbe(torch.nn.Module):
    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        for name, tensor in weights.items():
            self.register_buffer(name, tensor)
        inv_freq = 1.0 / (
            ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
        )
        self.register_buffer("inv_freq", inv_freq.contiguous())
        self.register_buffer(
            "physical_slots", torch.arange(PHYSICAL_BLOCKS * BLOCK_SIZE, dtype=torch.int32)
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
            self.inv_freq.view(1, HEAD_DIM // 2, 1), position.float().view(1, 1, 1)
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
        hidden: torch.Tensor,
        position: torch.Tensor,
        sequence_length: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        explicit_tiling: torch.Tensor,
    ):
        residual = hidden
        input_norm = self.rms_norm(hidden, self.input_norm)
        query = F.linear(input_norm, self.q_weight, self.q_bias)
        key = F.linear(input_norm, self.k_weight, self.k_bias)
        value = F.linear(input_norm, self.v_weight, self.v_bias)
        query = query.view(1, 1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        key = key.view(1, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        value = value.view(1, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        cos, sin = self.rope(position, hidden.dtype)
        cos = cos.view(1, 1, 1, HEAD_DIM)
        sin = sin.view(1, 1, 1, HEAD_DIM)
        query_rope = query * cos + self.rotate_half(query) * sin
        key_rope = key * cos + self.rotate_half(key) * sin

        updated_key, dense_key = self.update_and_read_cache(
            key_cache, key_rope, block_table, slot_mapping
        )
        updated_value, dense_value = self.update_and_read_cache(
            value_cache, value, block_table, slot_mapping
        )
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
            query_rope.squeeze(2).contiguous(),
            expanded_key.transpose(1, 2).contiguous(),
            explicit_tiling,
        )
        scaled_scores = (raw_qk.float() / math.sqrt(HEAD_DIM)).to(torch.bfloat16).unsqueeze(2)
        scores = torch.ops.g4a_barrier.materialize(scaled_scores)
        valid = self.logical_offsets < sequence_length.reshape(1)
        masked_scores = torch.where(
            valid.view(1, 1, 1, LOGICAL_CAPACITY),
            scores,
            torch.full_like(scores, torch.finfo(torch.bfloat16).min),
        )
        probabilities = torch.softmax(masked_scores.float(), dim=-1).to(torch.bfloat16)
        attention_value = torch.matmul(probabilities, expanded_value)
        attention_flat = attention_value.transpose(1, 2).reshape(1, 1, HIDDEN_SIZE)
        attention_projection = F.linear(attention_flat, self.o_weight)
        hidden_after_attention = residual + attention_projection

        post_attention_norm = self.rms_norm(hidden_after_attention, self.post_norm)
        gate = torch.ops.g4a_materialize.materialize(
            F.silu(F.linear(post_attention_norm, self.gate_weight))
        )
        up = torch.ops.g4a_materialize.materialize(
            F.linear(post_attention_norm, self.up_weight)
        )
        mlp_product = gate * up
        mlp_projection = F.linear(mlp_product, self.down_weight)
        hidden_after_mlp = hidden_after_attention + mlp_projection
        return (
            updated_key,
            updated_value,
            position + 1,
            input_norm,
            query_rope,
            key_rope,
            value,
            masked_scores,
            probabilities,
            attention_value,
            attention_projection,
            hidden_after_attention,
            post_attention_norm,
            gate,
            up,
            mlp_product,
            mlp_projection,
            hidden_after_mlp,
        )


def register_custom_ops(
    exact_qk_source: Path, barrier_source: Path, materialize_source: Path
) -> None:
    sys.path.insert(0, str(exact_qk_source))
    sys.path.insert(0, str(barrier_source))
    sys.path.insert(0, str(materialize_source))
    import export_qk_attempt50_air  # noqa: F401
    import export_barrier_probe  # noqa: F401
    import export_probe  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--full-reference", type=Path, required=True)
    parser.add_argument("--exact-qk-source", type=Path, required=True)
    parser.add_argument("--barrier-source", type=Path, required=True)
    parser.add_argument("--materialize-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    register_custom_ops(args.exact_qk_source, args.barrier_source, args.materialize_source)
    import torch_npu  # noqa: F401
    from torchair.configs.compiler_config import CompilerConfig
    from torchair.npu_export import dynamo_export

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    full = np.load(args.full_reference)
    token_ids = full["token_ids"]
    weights, hidden_rows = load_layer0(args.model_dir, token_ids)
    model = Layer0BoundaryProbe(weights).eval().npu()
    hidden_rows_npu = hidden_rows.npu()
    key = bits_to_bf16(full["input_key_cache_bits"][0]).npu()
    value = bits_to_bf16(full["input_value_cache_bits"][0]).npu()
    block_table = torch.from_numpy(full["block_table"].copy()).npu()
    tiling = torch.from_numpy(full["tiling"].copy()).npu()
    reference = {
        "token_ids": token_ids,
        "hidden_inputs_bits": bf16_bits(hidden_rows),
        "input_key_cache_bits": bf16_bits(key),
        "input_value_cache_bits": bf16_bits(value),
        "block_table": full["block_table"],
        "tiling": full["tiling"],
    }
    storage = {}
    eager_cache_exact = {}
    all_finite = True
    with torch.no_grad():
        for step in range(1, 5):
            hidden = hidden_rows_npu[step - 1 : step]
            position = torch.tensor([step - 1], dtype=torch.int64, device="npu")
            sequence_length = torch.tensor([[step]], dtype=torch.int32, device="npu")
            slot_mapping = torch.tensor([BLOCK_SIZE + step - 1], dtype=torch.int32, device="npu")
            outputs = model(
                hidden, position, sequence_length, block_table, slot_mapping, key, value, tiling
            )
            for name, tensor in zip(OUTPUT_NAMES, outputs):
                if tensor.dtype == torch.bfloat16:
                    array = bf16_bits(tensor)
                    all_finite = all_finite and bool(torch.isfinite(tensor.float()).all().item())
                    reference[f"step{step}_{name}_bits"] = array
                    storage[name] = {
                        "abi_dtype": "DT_BF16",
                        "storage_dtype": "uint16",
                        "shape": list(array.shape),
                        "bytes": int(array.nbytes),
                    }
                elif tensor.dtype == torch.int64:
                    array = tensor.detach().contiguous().cpu().numpy()
                    reference[f"step{step}_{name}"] = array
                    storage[name] = {
                        "abi_dtype": "DT_INT64",
                        "storage_dtype": "int64",
                        "shape": list(array.shape),
                        "bytes": int(array.nbytes),
                    }
                else:
                    raise RuntimeError(f"unexpected output dtype for {name}: {tensor.dtype}")
            expected_key = full[f"step{step}_key_cache_bits"][0]
            expected_value = full[f"step{step}_value_cache_bits"][0]
            eager_cache_exact[str(step)] = {
                "key": bool(np.array_equal(reference[f"step{step}_updated_key_bits"], expected_key)),
                "value": bool(
                    np.array_equal(reference[f"step{step}_updated_value_bits"], expected_value)
                ),
            }
            key, value = outputs[0], outputs[1]

    reference_path = args.output_dir / "attempt55d-eager-reference.npz"
    np.savez(reference_path, **reference)
    eager_pass = all_finite and all(
        all(checks.values()) for checks in eager_cache_exact.values()
    )
    eager_screen = {
        "pass": eager_pass,
        "all_outputs_finite": all_finite,
        "layer0_cache_exact_to_full_attempt53k": eager_cache_exact,
        "reference_sha256": sha256(reference_path),
        "output_specs": [dict(name=name, **storage[name]) for name in OUTPUT_NAMES],
    }
    (args.output_dir / "eager-screen.json").write_text(
        json.dumps(eager_screen, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    if not eager_pass:
        print(json.dumps(eager_screen, indent=2), flush=True)
        raise SystemExit(10)

    initial_hidden = hidden_rows_npu[:1]
    initial_position = torch.zeros((1,), dtype=torch.int64, device="npu")
    initial_length = torch.ones((1, 1), dtype=torch.int32, device="npu")
    initial_slot = torch.tensor([BLOCK_SIZE], dtype=torch.int32, device="npu")
    initial_key = bits_to_bf16(full["input_key_cache_bits"][0]).npu()
    initial_value = bits_to_bf16(full["input_value_cache_bits"][0]).npu()
    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        initial_hidden,
        initial_position,
        initial_length,
        block_table,
        initial_slot,
        initial_key,
        initial_value,
        tiling,
        model=model,
        export_path=str(args.output_dir),
        export_name="qwen_layer0_boundary_attempt55d",
        dynamic=False,
        config=config,
    )
    air = args.output_dir / "qwen_layer0_boundary_attempt55d.air"
    graph = args.output_dir / "dynamo.pbtxt"
    external = [path for path in args.output_dir.iterdir() if path.is_file() and path.suffix == ""]
    export_result = {
        "pass": air.is_file() and graph.is_file(),
        "air_sha256": sha256(air) if air.is_file() else None,
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "external_file_count": len(external),
        "external_file_bytes": sum(path.stat().st_size for path in external),
        "reference_sha256": sha256(reference_path),
        "eager_pass": eager_pass,
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(export_result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"eager": eager_screen, "export": export_result}, indent=2), flush=True)
    if not export_result["pass"]:
        raise SystemExit(11)


if __name__ == "__main__":
    main()
