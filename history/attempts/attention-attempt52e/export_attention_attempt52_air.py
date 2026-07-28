#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from torchair.configs.compiler_config import CompilerConfig
from torchair.npu_export import dynamo_export


ATTEMPT7_OUTPUT_NAMES = (
    "attention",
    "key_cache",
    "value_cache",
    "position",
    "k_projection",
    "k_rope",
    "qk_scores",
    "q_projection",
    "q_rope",
    "rope_cos",
    "rope_sin",
    "q_projection_bf16",
    "k_projection_bf16",
    "q_rope_bf16",
    "k_rope_bf16",
    "updated_key_bf16",
)
OUTPUT_NAMES = ATTEMPT7_OUTPUT_NAMES + ("qk_scores_bf16",)
TILING_WORDS = (28, 1, 128, 8, 16, 512, 16, 1, 1, 1, 28, 5, 2336, 24, 0, 0, 0, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_array(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.cpu().numpy()


class RepairedQwenAttentionSlice(torch.nn.Module):
    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        for name, tensor in base.named_buffers():
            self.register_buffer(name, tensor.detach().clone().contiguous())

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        head_dim = x.shape[-1]
        return torch.cat((-x[..., head_dim // 2 :], x[..., : head_dim // 2]), dim=-1)

    def forward(self, hidden_table, key_cache, value_cache, position, explicit_tiling):
        hidden_size = self.q_weight.shape[1]
        num_heads = self.q_weight.shape[0] // 128
        num_kv_heads = self.k_weight.shape[0] // 128
        num_kv_groups = num_heads // num_kv_heads
        max_kv = key_cache.shape[2]

        hidden = torch.index_select(hidden_table, 0, position).view(1, 1, hidden_size)
        hidden_bf16 = hidden.to(torch.bfloat16)
        variance = hidden_bf16.float().pow(2).mean(dim=-1, keepdim=True)
        normed = hidden_bf16 * torch.rsqrt(variance + 1e-6).to(torch.bfloat16)
        normed = normed * self.norm_weight

        q_projection = F.linear(normed, self.q_weight, self.q_bias)
        k_projection = F.linear(normed, self.k_weight, self.k_bias)
        v_projection = F.linear(normed, self.v_weight, self.v_bias)
        query = q_projection.view(1, 1, num_heads, 128).transpose(1, 2)
        key = k_projection.view(1, 1, num_kv_heads, 128).transpose(1, 2)
        value = v_projection.view(1, 1, num_kv_heads, 128).transpose(1, 2)

        cos = torch.index_select(self.rope_cos, 0, position).view(1, 1, 1, 128)
        sin = torch.index_select(self.rope_sin, 0, position).view(1, 1, 1, 128)
        q_rope = query * cos + self.rotate_half(query) * sin
        k_rope = key * cos + self.rotate_half(key) * sin

        key_cache_bf16 = key_cache.to(torch.bfloat16)
        value_cache_bf16 = value_cache.to(torch.bfloat16)
        slot_mask = (self.cache_slots == position.view(1, 1, 1, 1)).to(torch.bfloat16)
        inverse_slot_mask = 1.0 - slot_mask
        updated_key = key_cache_bf16 * inverse_slot_mask + k_rope * slot_mask
        updated_value = value_cache_bf16 * inverse_slot_mask + value * slot_mask

        expanded_key = (
            updated_key.unsqueeze(2)
            .expand(1, num_kv_heads, num_kv_groups, max_kv, 128)
            .reshape(1, num_heads, max_kv, 128)
        )
        expanded_value = (
            updated_value.unsqueeze(2)
            .expand(1, num_kv_heads, num_kv_groups, max_kv, 128)
            .reshape(1, num_heads, max_kv, 128)
        )
        qk_raw = torch.ops.g4a_qk.exact_qk(
            q_rope.squeeze(2).contiguous(),
            expanded_key.squeeze(0).transpose(1, 2).contiguous(),
            explicit_tiling,
        )
        scaled_qk = (qk_raw.float() / math.sqrt(128)).to(torch.bfloat16).unsqueeze(2)
        qk_scores = torch.ops.g4a_barrier.materialize(scaled_qk)
        valid = self.cache_slots.squeeze(-1) <= position.view(1, 1, 1)
        masked_scores = torch.where(
            valid.unsqueeze(2), qk_scores, torch.full_like(qk_scores, -10000.0)
        )
        probabilities_bf16 = torch.softmax(masked_scores.float(), dim=-1).to(torch.bfloat16)
        probabilities = torch.ops.g4a_barrier.materialize(probabilities_bf16)
        attention_value = torch.matmul(probabilities, expanded_value)
        attention_flat = attention_value.transpose(1, 2).reshape(1, 1, hidden_size)
        output = F.linear(attention_flat, self.o_weight)

        return (
            output.to(torch.float16),
            updated_key.to(torch.float16),
            updated_value.to(torch.float16),
            position + 1,
            k_projection.to(torch.float16),
            k_rope.to(torch.float16),
            qk_scores.to(torch.float32),
            q_projection.to(torch.float16),
            q_rope.to(torch.float16),
            cos.to(torch.float16),
            sin.to(torch.float16),
            q_projection,
            k_projection,
            q_rope,
            k_rope,
            updated_key,
            qk_scores,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2d-root", type=Path, required=True)
    parser.add_argument("--g2e-root", type=Path, required=True)
    parser.add_argument("--custom-source", type=Path, required=True)
    parser.add_argument("--barrier-source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--original-reference", type=Path, required=True)
    parser.add_argument("--frozen-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eager-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.g2e_root))
    sys.path.insert(0, str(args.custom_source))
    sys.path.insert(0, str(args.barrier_source))
    sys.path.insert(0, str(args.g2d_root))
    import export_qk_attempt50_air  # noqa: F401,E402; registers g4a_qk
    import export_barrier_probe  # noqa: F401,E402; registers g4a_barrier
    from export_qwen_attention_air import QwenAttentionKvSlice, load_inputs_and_weights  # noqa: E402

    torch.npu.set_device(0)
    weights, hidden_table, token_ids, tokens, _ = load_inputs_and_weights(args.model_dir)
    base = QwenAttentionKvSlice(weights).eval().npu()
    model = RepairedQwenAttentionSlice(base).eval().npu()
    hidden_npu = hidden_table.npu()
    original = np.load(args.original_reference)
    frozen = np.load(args.frozen_reference)
    tiling_np = np.asarray(TILING_WORDS, dtype="<u4").view(np.uint8).copy()
    tiling_npu = torch.from_numpy(tiling_np).npu()
    tiling_path = args.output_dir / "tiling.bin"
    tiling_np.tofile(tiling_path)

    key = torch.from_numpy(original["input_key_cache"]).npu()
    value = torch.from_numpy(original["input_value_cache"]).npu()
    position = torch.from_numpy(original["input_position"]).npu()
    reference = {}
    exact_by_step = {}
    metrics_by_step = {}
    with torch.no_grad():
        for step in range(1, 5):
            outputs = model(hidden_npu, key, value, position, tiling_npu)
            arrays = [to_array(output) for output in outputs]
            exact_by_step[str(step)] = {
                name: bool(np.array_equal(arrays[index], frozen[f"step{step}_{name}"]))
                for index, name in enumerate(ATTEMPT7_OUTPUT_NAMES)
            }
            metrics_by_step[str(step)] = {}
            for index, name in enumerate(ATTEMPT7_OUTPUT_NAMES):
                actual = arrays[index]
                expected = frozen[f"step{step}_{name}"]
                difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
                metrics_by_step[str(step)][name] = {
                    "all_exact": exact_by_step[str(step)][name],
                    "max_abs_error": float(np.max(difference)),
                    "mismatch_count": int(actual.size - np.count_nonzero(np.equal(actual, expected))),
                }
            for name, array in zip(OUTPUT_NAMES, arrays):
                reference[f"step{step}_{name}"] = array
            key, value, position = outputs[1:4]
    eager_exact = all(all(checks.values()) for checks in exact_by_step.values())
    reference_path = args.output_dir / "attempt52-eager-reference.npz"
    np.savez(reference_path, **reference)
    eager_screen = {
        "eager_first_sixteen_exact_to_frozen_attempt7": eager_exact,
        "exact_by_step": exact_by_step,
        "metrics_by_step": metrics_by_step,
        "reference_sha256": sha256(reference_path),
    }
    (args.output_dir / "eager-screen.json").write_text(
        json.dumps(eager_screen, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("G4A_ATTEMPT52_EAGER_SCREEN " + json.dumps(eager_screen, ensure_ascii=True), flush=True)
    if not eager_exact:
        raise RuntimeError("repaired eager slice differs from frozen Attempt 7")
    if args.eager_only:
        return

    initial_key = torch.from_numpy(original["input_key_cache"]).npu()
    initial_value = torch.from_numpy(original["input_value_cache"]).npu()
    initial_position = torch.from_numpy(original["input_position"]).npu()
    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        hidden_npu,
        initial_key,
        initial_value,
        initial_position,
        tiling_npu,
        model=model,
        export_path=str(args.output_dir),
        export_name="qwen_attention_attempt52",
        dynamic=False,
        config=config,
    )
    air = args.output_dir / "qwen_attention_attempt52.air"
    graph = args.output_dir / "dynamo.pbtxt"
    result = {
        "execution_success": air.is_file() and graph.is_file(),
        "candidate": "custom_raw_qk_with_qk_and_softmax_opaque_bf16_barriers",
        "token_ids": token_ids,
        "tokens": tokens,
        "output_names": list(OUTPUT_NAMES),
        "eager_first_sixteen_exact_to_frozen_attempt7": eager_exact,
        "exact_by_step": exact_by_step,
        "metrics_by_step": metrics_by_step,
        "model_dir": str(args.model_dir),
        "reference_sha256": sha256(reference_path),
        "tiling_sha256": sha256(tiling_path),
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("G4A_ATTEMPT52_EXPORT " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["execution_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
