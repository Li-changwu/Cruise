#!/usr/bin/env python3
"""Export and validate a fixed-K B=4 resident decoder epoch AIR."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).parents[2]
for relative in (
    "history/attempts/g4c-attempt69c-b4-air",
    "history/attempts/qk-attempt50",
    "history/attempts/bf16-barrier",
    "history/attempts/bf16-materialize-attempt56r1",
):
    sys.path.insert(0, str(ROOT / relative))

from batched_decoder_step import (  # noqa: E402
    BATCH_SIZE,
    BLOCK_SIZE,
    HEAD_DIM,
    LOGICAL_CAPACITY,
    NUM_KV_HEADS,
    NUM_LAYERS,
    TILING_WORDS,
    PagedQwenDecoderStep,
    load_checkpoint,
    register_custom_ops,
)


K6_PHYSICAL_BLOCKS = BATCH_SIZE


class FixedLayoutPagedQwenDecoderStep(PagedQwenDecoderStep):
    """Decoder specialization for the fixed K=6 per-row KV allocation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # One physical block belongs to each fixed batch row. Keeping all
        # physical slots as a buffer lets GE lower row updates as static
        # SelectV2 kernels instead of dynamic whole-cache ScatterElements.
        # A single flat cache chain also avoids a per-row stack materializing
        # between decoder layers.
        self.register_buffer(
            "fixed_physical_slots",
            torch.arange(BATCH_SIZE * BLOCK_SIZE, dtype=torch.int32),
        )

    def update_and_read_cache(
        self,
        layer_cache: torch.Tensor,
        new_value: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # K=6 assigns one physical block to each fixed batch row. The static
        # row layout avoids dynamic scatter lowering while touching only the
        # row which owns a token. ``block_table`` remains part of the public
        # decoder ABI and is used by the epoch wrapper when calculating the
        # next slot.
        del block_table
        flat = layer_cache.reshape(
            BATCH_SIZE * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
        )
        safe_slots = slot_mapping.clamp_min(0)
        updated_flat = flat
        for batch_index in range(BATCH_SIZE):
            update_mask = (
                self.fixed_physical_slots == safe_slots[batch_index : batch_index + 1]
            ).view(BATCH_SIZE * BLOCK_SIZE, 1, 1)
            update_mask = update_mask & active_mask[
                batch_index : batch_index + 1
            ].to(torch.bool).view(1, 1, 1)
            replacement = new_value[batch_index].reshape(
                1, NUM_KV_HEADS, HEAD_DIM
            )
            updated_flat = torch.where(update_mask, replacement, updated_flat)
        dense = updated_flat.reshape(
            BATCH_SIZE, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
        )[:, :LOGICAL_CAPACITY]
        return (
            updated_flat.reshape_as(layer_cache),
            dense.permute(0, 2, 1, 3).contiguous(),
        )


class FixedKEpochDecoder(torch.nn.Module):
    """Unroll greedy decode on Device while preserving the decoder input ABI."""

    def __init__(self, decoder: PagedQwenDecoderStep, epoch_steps: int) -> None:
        super().__init__()
        self.decoder = decoder
        self.epoch_steps = epoch_steps

    @staticmethod
    def _next_slot(block_table: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        # The resident benchmark's logical capacity is smaller than one block,
        # but keep the block-table lookup valid for every fixed graph input.
        logical_block = torch.div(position, BLOCK_SIZE, rounding_mode="floor")
        safe_block = logical_block.clamp(0, block_table.shape[1] - 1).to(torch.int64)
        physical_block = torch.gather(block_table, 1, safe_block.view(BATCH_SIZE, 1))
        slot = physical_block.view(BATCH_SIZE) * BLOCK_SIZE + (position % BLOCK_SIZE).to(
            torch.int32
        )
        return torch.where(
            position < LOGICAL_CAPACITY,
            slot,
            torch.full_like(slot, -1),
        )

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
        active_mask: torch.Tensor,
        eos_token_ids: torch.Tensor,
        step_limit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history: list[torch.Tensor] = []
        current_token = token_id
        current_position = position
        current_length = sequence_length
        current_slot = slot_mapping
        current_key = key_cache
        current_value = value_cache
        current_active = active_mask
        for step in range(self.epoch_steps):
            step_index = torch.full_like(step_limit.reshape(1), step)
            run_active = torch.where(
                step_limit.reshape(1) > step_index,
                current_active,
                torch.zeros_like(current_active),
            )
            logits, current_key, current_value, next_position = self.decoder(
                current_token,
                current_position,
                current_length,
                block_table,
                current_slot,
                current_key,
                current_value,
                explicit_tiling,
                run_active,
            )
            generated = torch.argmax(logits, dim=-1)
            generated_flat = generated.reshape(BATCH_SIZE)
            history.append(
                torch.where(
                    run_active.to(torch.bool),
                    generated_flat,
                    torch.full_like(generated_flat, -1),
                )
            )
            current_token = torch.where(
                run_active.to(torch.bool).view(BATCH_SIZE, 1),
                generated.view(BATCH_SIZE, 1),
                current_token,
            )
            current_length = torch.where(
                run_active.to(torch.bool).view(BATCH_SIZE, 1),
                (next_position + 1).to(torch.int32).view(BATCH_SIZE, 1),
                current_length,
            )
            current_slot = self._next_slot(block_table, next_position)
            current_position = next_position
            current_active = torch.where(
                run_active.to(torch.bool),
                (generated_flat != eos_token_ids).to(torch.int32),
                current_active,
            )
        return torch.stack(history), current_key, current_value, current_position


def _inputs() -> tuple[torch.Tensor, ...]:
    token = torch.full((BATCH_SIZE, 1), 9707, dtype=torch.int64, device="npu")
    position = torch.zeros((BATCH_SIZE,), dtype=torch.int64, device="npu")
    sequence_length = torch.ones((BATCH_SIZE, 1), dtype=torch.int32, device="npu")
    block_table = torch.tensor(
        [[0], [1], [2], [3]], dtype=torch.int32, device="npu"
    )
    slot_mapping = torch.tensor(
        [0, BLOCK_SIZE, 2 * BLOCK_SIZE, 3 * BLOCK_SIZE],
        dtype=torch.int32,
        device="npu",
    )
    cache_shape = (
        NUM_LAYERS,
        K6_PHYSICAL_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    key_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device="npu")
    value_cache = torch.zeros(cache_shape, dtype=torch.bfloat16, device="npu")
    tiling = torch.from_numpy(
        np.asarray(TILING_WORDS, dtype="<u4").view(np.uint8).copy()
    ).npu()
    active_mask = torch.tensor([1, 1, 0, 1], dtype=torch.int32, device="npu")
    eos_token_ids = torch.full(
        (BATCH_SIZE,), 151645, dtype=torch.int64, device="npu"
    )
    step_limit = torch.tensor([6], dtype=torch.int32, device="npu")
    return (
        token,
        position,
        sequence_length,
        block_table,
        slot_mapping,
        key_cache,
        value_cache,
        tiling,
        active_mask,
        eos_token_ids,
        step_limit,
    )


def _reference(
    decoder: PagedQwenDecoderStep, inputs: tuple[torch.Tensor, ...], epoch_steps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        current_token,
        current_position,
        current_length,
        block_table,
        current_slot,
        current_key,
        current_value,
        tiling,
        current_active,
        eos_token_ids,
        step_limit,
    ) = inputs
    history: list[torch.Tensor] = []
    for step in range(epoch_steps):
        step_index = torch.full_like(step_limit.reshape(1), step)
        run_active = torch.where(
            step_limit.reshape(1) > step_index,
            current_active,
            torch.zeros_like(current_active),
        )
        logits, current_key, current_value, next_position = decoder(
            current_token,
            current_position,
            current_length,
            block_table,
            current_slot,
            current_key,
            current_value,
            tiling,
            run_active,
        )
        generated = torch.argmax(logits, dim=-1)
        generated_flat = generated.reshape(BATCH_SIZE)
        history.append(
            torch.where(
                run_active.to(torch.bool),
                generated_flat,
                torch.full_like(generated_flat, -1),
            )
        )
        current_token = torch.where(
            run_active.to(torch.bool).view(BATCH_SIZE, 1),
            generated.view(BATCH_SIZE, 1),
            current_token,
        )
        current_length = torch.where(
            run_active.to(torch.bool).view(BATCH_SIZE, 1),
            (next_position + 1).to(torch.int32).view(BATCH_SIZE, 1),
            current_length,
        )
        current_slot = FixedKEpochDecoder._next_slot(block_table, next_position)
        current_position = next_position
        current_active = torch.where(
            run_active.to(torch.bool),
            (generated_flat != eos_token_ids).to(torch.int32),
            current_active,
        )
    return torch.stack(history), current_key, current_value, current_position


def _install_custom_opp(asset_root: Path) -> tuple[str, ...]:
    vendors = (
        asset_root / "custom-opp/install-attempt47/vendors/vllm-ascend",
        asset_root / "custom-opp/install-attempt69a-b4-barrier/vendors/vllm-ascend",
        asset_root / "custom-opp/install-attempt56r1/vendors/vllm-ascend",
    )
    for vendor in vendors:
        if not (vendor / "op_impl").is_dir() or not (vendor / "op_api/lib").is_dir():
            raise RuntimeError(f"incomplete custom OPP vendor: {vendor}")
    custom_opp = ":".join(str(vendor) for vendor in vendors)
    custom_libraries = ":".join(str(vendor / "op_api/lib") for vendor in vendors)
    inherited_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    inherited_libraries = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        value for value in (custom_opp, inherited_opp) if value
    )
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        value for value in (custom_libraries, inherited_libraries) if value
    )
    return tuple(str(vendor) for vendor in vendors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--epoch-steps", type=int, default=6)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--eager-check", action="store_true")
    args = parser.parse_args()
    if args.epoch_steps != 6:
        raise ValueError("the M4b epoch AIR is intentionally fixed at K=6")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    custom_opp_vendors = _install_custom_opp(args.asset_root.resolve(strict=True))

    register_custom_ops(
        ROOT / "history/attempts/qk-attempt50",
        ROOT / "history/attempts/bf16-barrier",
        ROOT / "history/attempts/bf16-materialize-attempt56r1",
    )
    import torch_npu  # noqa: F401
    from torchair.configs.compiler_config import CompilerConfig
    from torchair.npu_export import dynamo_export

    torch.npu.set_device(0)
    decoder = FixedLayoutPagedQwenDecoderStep(
        load_checkpoint(args.model_dir),
        batch_size=BATCH_SIZE,
        physical_blocks=K6_PHYSICAL_BLOCKS,
    ).eval().npu()
    model = FixedKEpochDecoder(decoder, args.epoch_steps).eval().npu()
    inputs = _inputs()
    checks: list[bool] | None = None
    if args.eager_check:
        with torch.no_grad():
            reference = _reference(decoder, inputs, args.epoch_steps)
            actual = model(*inputs)
            torch.npu.synchronize()
        checks = [
            bool(torch.equal(value, expected)) for value, expected in zip(actual, reference)
        ]
        if not all(checks):
            raise RuntimeError(f"fixed-K eager recurrence mismatch: {checks}")

    if not args.skip_export:
        config = CompilerConfig()
        config.mode = "max-autotune"
        dynamo_export(
            *inputs,
            model=model,
            export_path=str(args.output_dir),
            export_name="qwen_b4_epoch_k6_argmax",
            dynamic=False,
            config=config,
        )
    air = args.output_dir / "qwen_b4_epoch_k6_argmax.air"
    result = {
        "gate": "M4b fixed-K Device decoder epoch export",
        "epoch_steps": args.epoch_steps,
        "custom_opp_vendors": custom_opp_vendors,
        "eager_recurrence_exact": checks,
        "eager_check_status": (
            "passed" if args.eager_check else "not_run_requires_private_eager_kernel"
        ),
        "export_skipped": args.skip_export,
        "air_exists": air.is_file(),
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "claim_boundary": (
            "AIR export only unless --eager-check is selected. Native GE "
            "recurrence and end-to-end scheduler semantics require separate "
            "validation."
        ),
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    if not args.skip_export and not result["air_exists"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
