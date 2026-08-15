#!/usr/bin/env python3
"""Export the P3 one-step greedy Qwen decoder with a 384-token KV lease."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


ROOT = Path(__file__).parents[2]
DECODER_SOURCE = ROOT / "history/attempts/g4c-attempt69c-b4-air"
sys.path.insert(0, str(DECODER_SOURCE))

import batched_decoder_step as base  # noqa: E402

from batched_decoder_step import (  # noqa: E402
    BATCH_SIZE,
    BLOCK_SIZE,
    HEAD_DIM,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    PagedQwenDecoderStep,
    load_checkpoint,
)


PHYSICAL_BLOCKS_PER_REQUEST = 3
PHYSICAL_BLOCKS = BATCH_SIZE * PHYSICAL_BLOCKS_PER_REQUEST
LOGICAL_CAPACITY = 384
VOCAB_SIZE = 152064
TILING_WORDS = (
    28,
    1,
    128,
    384,
    16,
    512,
    128,
    1,
    1,
    3,
    84,
    5,
    2336,
    24,
    0,
    0,
    0,
    0,
)
STOCK_VLLM_TOKEN_ORACLE = {
    (1000, 1001, 1002, 1003): (355, 11, 220, 17, 15, 16, 16, 11),
    tuple(range(1000, 1128)): (11, 220, 16, 15, 15, 15, 15, 15),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The existing cache update implementation is page-generic. Its forward path
# has an eight-token shape constant, so the P3 subclass below owns that path.
base.LOGICAL_CAPACITY = LOGICAL_CAPACITY


class P3PagedQwenDecoderStep(PagedQwenDecoderStep):
    """One decoder token over four rows and three physical pages per row."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # vLLM executes QKV and gate/up as merged projections.  Keep those
        # exact GEMM boundaries in the exported graph; separate BF16 GEMMs can
        # select different greedy tokens even though the real-valued formulas
        # are equivalent.
        for layer in self.layers:
            qkv_weight = torch.cat(
                (layer.q_weight, layer.k_weight, layer.v_weight), dim=0
            ).contiguous()
            qkv_bias = torch.cat(
                (layer.q_bias, layer.k_bias, layer.v_bias), dim=0
            ).contiguous()
            gate_up_weight = torch.cat(
                (layer.gate_weight, layer.up_weight), dim=0
            ).contiguous()
            for name in (
                "q_weight",
                "q_bias",
                "k_weight",
                "k_bias",
                "v_weight",
                "v_bias",
                "gate_weight",
                "up_weight",
            ):
                delattr(layer, name)
            layer.register_buffer("qkv_weight", qkv_weight)
            layer.register_buffer("qkv_bias", qkv_bias)
            layer.register_buffer("gate_up_weight", gate_up_weight)

    @staticmethod
    def npu_rms_norm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(hidden, weight, epsilon=base.RMS_EPS)[0]

    @staticmethod
    def npu_add_rms_norm(
        hidden: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, _, combined = torch_npu.npu_add_rms_norm(
            hidden, residual, weight, base.RMS_EPS
        )
        return normalized, combined

    def forward(
        self,
        token_id: torch.Tensor,
        position: torch.Tensor,
        sequence_length: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.embedding(token_id, self.embedding).view(
            self.batch_size, 1, base.HIDDEN_SIZE
        )
        cos, sin = self.rope(position, hidden.dtype)
        cos = cos.view(self.batch_size, 1, 1, HEAD_DIM)
        sin = sin.view(self.batch_size, 1, 1, HEAD_DIM)
        valid = self.logical_offsets.view(1, LOGICAL_CAPACITY) < sequence_length.reshape(
            self.batch_size, 1
        )
        new_key_layers = []
        new_value_layers = []
        residual = None

        for layer_index, layer in enumerate(self.layers):
            if residual is None:
                residual = hidden
                normalized = self.npu_rms_norm(hidden, layer.input_norm)
            else:
                normalized, residual = self.npu_add_rms_norm(
                    hidden, residual, layer.input_norm
                )
            normalized_2d = normalized.reshape(-1, base.HIDDEN_SIZE)
            qkv = F.linear(normalized_2d, layer.qkv_weight, layer.qkv_bias)
            query, key, value = qkv.split(
                (base.HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM),
                dim=-1,
            )
            query = query.reshape(self.batch_size, 1, base.HIDDEN_SIZE)
            key = key.reshape(self.batch_size, 1, NUM_KV_HEADS * HEAD_DIM)
            value = value.reshape(self.batch_size, 1, NUM_KV_HEADS * HEAD_DIM)
            query = query.view(self.batch_size, 1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            key = key.view(self.batch_size, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            value = value.view(self.batch_size, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            query = torch_npu.npu_rotary_mul(query, cos, sin)
            key = torch_npu.npu_rotary_mul(key, cos, sin)

            updated_key, dense_key = self.update_and_read_cache(
                key_cache[layer_index], key, block_table, slot_mapping, active_mask
            )
            updated_value, dense_value = self.update_and_read_cache(
                value_cache[layer_index], value, block_table, slot_mapping, active_mask
            )
            new_key_layers.append(updated_key)
            new_value_layers.append(updated_value)
            attention_mask = (~valid).view(
                self.batch_size, 1, 1, LOGICAL_CAPACITY
            )
            attention = torch_npu.npu_fused_infer_attention_score(
                query,
                dense_key,
                dense_value,
                num_heads=NUM_HEADS,
                num_key_value_heads=NUM_KV_HEADS,
                input_layout="BNSD",
                atten_mask=attention_mask,
                scale=HEAD_DIM**-0.5,
                sparse_mode=0,
            )[0]
            attention = attention.transpose(1, 2).reshape(
                self.batch_size, 1, base.HIDDEN_SIZE
            )
            attention_projection = F.linear(
                attention.reshape(-1, base.HIDDEN_SIZE), layer.o_weight
            ).reshape(self.batch_size, 1, base.HIDDEN_SIZE)
            normalized, residual = self.npu_add_rms_norm(
                attention_projection, residual, layer.post_norm
            )
            normalized_2d = normalized.reshape(-1, base.HIDDEN_SIZE)
            gate, up = F.linear(
                normalized_2d, layer.gate_up_weight
            ).split(
                (base.INTERMEDIATE_SIZE, base.INTERMEDIATE_SIZE), dim=-1
            )
            gate = F.silu(gate)
            mlp_projection = F.linear(
                (gate * up).reshape(-1, base.INTERMEDIATE_SIZE), layer.down_weight
            ).reshape(self.batch_size, 1, base.HIDDEN_SIZE)
            hidden = mlp_projection

        if residual is None:
            raise RuntimeError("decoder has no layers")
        hidden, _ = self.npu_add_rms_norm(hidden, residual, self.final_norm)
        logits = F.linear(hidden.reshape(-1, base.HIDDEN_SIZE), self.lm_head).reshape(
            self.batch_size, 1, VOCAB_SIZE
        )
        token = torch.argmax(logits.float(), dim=-1).to(torch.int64)
        next_position = torch.where(active_mask.to(torch.bool), position + 1, position)
        return token, torch.stack(new_key_layers), torch.stack(new_value_layers), next_position


def decoder_inputs() -> tuple[torch.Tensor, ...]:
    token = torch.full((BATCH_SIZE, 1), 9707, dtype=torch.int64, device="npu")
    position = torch.zeros((BATCH_SIZE,), dtype=torch.int64, device="npu")
    sequence_length = torch.ones((BATCH_SIZE, 1), dtype=torch.int32, device="npu")
    block_table = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        dtype=torch.int32,
        device="npu",
    )
    slot_mapping = torch.tensor(
        [0, 3 * BLOCK_SIZE, 6 * BLOCK_SIZE, 9 * BLOCK_SIZE],
        dtype=torch.int32,
        device="npu",
    )
    cache_shape = (NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    key_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device="npu")
    value_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device="npu")
    active_mask = torch.ones((BATCH_SIZE,), dtype=torch.int32, device="npu")
    return (
        token,
        position,
        sequence_length,
        block_table,
        slot_mapping,
        key_cache,
        value_cache,
        active_mask,
    )


def next_slots(block_table: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    block = torch.div(positions, BLOCK_SIZE, rounding_mode="floor").to(torch.int64)
    physical = torch.gather(block_table, 1, block.view(BATCH_SIZE, 1)).view(BATCH_SIZE)
    return physical * BLOCK_SIZE + (positions % BLOCK_SIZE).to(torch.int32)


def eager_generate(
    model: P3PagedQwenDecoderStep, prompt: tuple[int, ...], output_tokens: int
) -> tuple[tuple[int, ...], ...]:
    inputs = list(decoder_inputs())
    generated: list[list[int]] = [[] for _ in range(BATCH_SIZE)]
    with torch.no_grad():
        output = None
        for prompt_index, prompt_token in enumerate(prompt):
            inputs[0] = torch.full(
                (BATCH_SIZE, 1), prompt_token, dtype=torch.int64, device="npu"
            )
            inputs[1] = torch.full(
                (BATCH_SIZE,), prompt_index, dtype=torch.int64, device="npu"
            )
            inputs[2] = torch.full(
                (BATCH_SIZE, 1),
                prompt_index + 1,
                dtype=torch.int32,
                device="npu",
            )
            inputs[4] = next_slots(inputs[3], inputs[1])
            output = model(*inputs)
            inputs[5] = output[1]
            inputs[6] = output[2]
        if output is None:
            raise ValueError("prompt must not be empty")
        for output_index in range(output_tokens):
            tokens = output[0].view(BATCH_SIZE).cpu().tolist()
            for row, token in enumerate(tokens):
                generated[row].append(int(token))
            if output_index + 1 == output_tokens:
                break
            inputs[0] = output[0]
            inputs[1] = output[3]
            inputs[2] = (output[3] + 1).to(torch.int32).view(BATCH_SIZE, 1)
            inputs[4] = next_slots(inputs[3], output[3])
            inputs[5] = output[1]
            inputs[6] = output[2]
            output = model(*inputs)
        torch.npu.synchronize()
    return tuple(tuple(row) for row in generated)


def eager_stock_semantics_check(model: P3PagedQwenDecoderStep) -> bool:
    for prompt, expected in STOCK_VLLM_TOKEN_ORACLE.items():
        observed = eager_generate(model, prompt, len(expected))
        if observed != (expected,) * BATCH_SIZE:
            return False
    return True


def eager_page_boundary_check(model: P3PagedQwenDecoderStep) -> bool:
    inputs = list(decoder_inputs())
    inputs[1] = torch.full((BATCH_SIZE,), 127, dtype=torch.int64, device="npu")
    inputs[2] = torch.full((BATCH_SIZE, 1), 128, dtype=torch.int32, device="npu")
    inputs[4] = torch.tensor(
        [127, 3 * BLOCK_SIZE + 127, 6 * BLOCK_SIZE + 127, 9 * BLOCK_SIZE + 127],
        dtype=torch.int32,
        device="npu",
    )
    def clone_inputs(values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [value.clone() for value in values]

    def same_outputs(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> bool:
        return len(left) == len(right) and all(
            torch.equal(actual, expected) for actual, expected in zip(left, right)
        )

    expected_output_shapes = (
        (BATCH_SIZE, 1),
        (NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        (NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        (BATCH_SIZE,),
    )
    with torch.no_grad():
        first = model(*inputs)
        torch.npu.synchronize()

        second_inputs = list(inputs)
        second_inputs[0] = first[0]
        second_inputs[1] = first[3]
        second_inputs[2] = (first[3] + 1).to(torch.int32).view(BATCH_SIZE, 1)
        second_inputs[4] = next_slots(second_inputs[3], first[3])
        second_inputs[5] = first[1]
        second_inputs[6] = first[2]
        replay_inputs = clone_inputs(second_inputs)
        second = model(*second_inputs)
        torch.npu.synchronize()
        replay = model(*replay_inputs)
        torch.npu.synchronize()
        reference = model(*inputs)
        torch.npu.synchronize()

    first_shapes_ok = tuple(value.shape for value in first) == expected_output_shapes
    second_shapes_ok = tuple(value.shape for value in second) == expected_output_shapes
    reference_exact = same_outputs(first, reference)
    replay_exact = same_outputs(second, replay)
    position_progression = bool(
        torch.equal(first[3].cpu(), torch.full((BATCH_SIZE,), 128, dtype=torch.int64))
        and torch.equal(second[3].cpu(), torch.full((BATCH_SIZE,), 129, dtype=torch.int64))
    )
    expected_boundary_blocks = torch.tensor([1, 4, 7, 10], dtype=torch.int64)
    expected_boundary_slots = (expected_boundary_blocks * BLOCK_SIZE).to(torch.int32)
    boundary_slot_ok = torch.equal(second_inputs[4].cpu(), expected_boundary_slots)
    # The page-1 slot must be written by the second step; this catches a
    # stale page cursor even when token/position outputs happen to match.
    key_boundary_written = bool(
        torch.any(
            second[1][:, expected_boundary_blocks, 0]
            != first[1][:, expected_boundary_blocks, 0]
        ).item()
    )
    return all(
        (
            first_shapes_ok,
            second_shapes_ok,
            reference_exact,
            replay_exact,
            position_progression,
            boundary_slot_ok,
            key_boundary_written,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eager-check", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    model = P3PagedQwenDecoderStep(
        load_checkpoint(args.model_dir),
        batch_size=BATCH_SIZE,
        physical_blocks=PHYSICAL_BLOCKS,
    ).eval().npu()
    eager = None
    eager_stock_semantics = None
    if args.eager_check:
        eager = eager_page_boundary_check(model)
        if not eager:
            raise RuntimeError("P3 page-boundary eager recurrence mismatch")
        eager_stock_semantics = eager_stock_semantics_check(model)
        if not eager_stock_semantics:
            raise RuntimeError("P3 eager tokens differ from the stock vLLM oracle")

    air = args.output_dir / "qwen_b4_p3_decoder_step.air"
    graph = args.output_dir / "dynamo.pbtxt"
    if not args.skip_export:
        from torchair.configs.compiler_config import CompilerConfig
        from torchair.npu_export import dynamo_export

        config = CompilerConfig()
        config.mode = "max-autotune"
        dynamo_export(
            *decoder_inputs(),
            model=model,
            export_path=str(args.output_dir),
            export_name="qwen_b4_p3_decoder_step",
            dynamic=False,
            config=config,
        )
        air = args.output_dir / "qwen_b4_p3_decoder_step.air"
        graph = args.output_dir / "dynamo.pbtxt"

    external_files = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.suffix == ""
    )

    result = {
        "gate": "P3-AIR384",
        "pass": bool(args.skip_export or (air.is_file() and graph.is_file())),
        "eager_page_boundary_exact": eager,
        "eager_stock_vllm_semantics_exact": eager_stock_semantics,
        "stock_vllm_token_oracle": [
            {"prompt": list(prompt), "tokens": list(tokens)}
            for prompt, tokens in STOCK_VLLM_TOKEN_ORACLE.items()
        ],
        "air_exists": air.is_file(),
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_exists": graph.is_file(),
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "external_file_count": len(external_files),
        "external_file_bytes": sum(path.stat().st_size for path in external_files),
        "physical_blocks": PHYSICAL_BLOCKS,
        "blocks_per_request": PHYSICAL_BLOCKS_PER_REQUEST,
        "logical_capacity": LOGICAL_CAPACITY,
        "cache_shape": [NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM],
        "input_abi": [
            ["DT_INT64", [BATCH_SIZE, 1]],
            ["DT_INT64", [BATCH_SIZE]],
            ["DT_INT32", [BATCH_SIZE, 1]],
            ["DT_INT32", [BATCH_SIZE, PHYSICAL_BLOCKS_PER_REQUEST]],
            ["DT_INT32", [BATCH_SIZE]],
            ["DT_BFLOAT16", [NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]],
            ["DT_BFLOAT16", [NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]],
            ["DT_INT32", [BATCH_SIZE]],
        ],
        "output_abi": [
            ["DT_INT64", [BATCH_SIZE, 1]],
            ["DT_BFLOAT16", [NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]],
            ["DT_BFLOAT16", [NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]],
            ["DT_INT64", [BATCH_SIZE]],
        ],
        "qk_probe_tiling_words": list(TILING_WORDS),
        "claim_boundary": (
            "P3 one-step AIR and page-boundary eager recurrence only; no "
            "Persistent Owner or service-performance claim."
        ),
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
