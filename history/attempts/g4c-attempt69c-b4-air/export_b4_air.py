#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from batched_decoder_step import (
    BATCH_SIZE,
    PHYSICAL_BLOCKS,
    PagedQwenDecoderStep,
    bits_to_bf16,
    load_checkpoint,
    register_custom_ops,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--exact-qk-source", type=Path, required=True)
    parser.add_argument("--barrier-source", type=Path, required=True)
    parser.add_argument("--materialize-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    register_custom_ops(
        args.exact_qk_source, args.barrier_source, args.materialize_source
    )
    import torch_npu  # noqa: F401
    from torchair.configs.compiler_config import CompilerConfig
    from torchair.npu_export import dynamo_export

    torch.npu.set_device(0)
    with np.load(args.reference) as reference:
        token = torch.from_numpy(reference["case0_token"].copy()).npu()
        position = torch.from_numpy(reference["case0_position"].copy()).npu()
        sequence_length = torch.from_numpy(
            reference["case0_sequence_length"].copy()
        ).npu()
        block_table = torch.from_numpy(reference["case0_block_table"].copy()).npu()
        slot_mapping = torch.from_numpy(
            reference["case0_slot_mapping"].copy()
        ).npu()
        key_cache = bits_to_bf16(
            reference["case0_input_key_cache_bits"].copy()
        ).npu()
        value_cache = bits_to_bf16(
            reference["case0_input_value_cache_bits"].copy()
        ).npu()
        explicit_tiling = torch.from_numpy(reference["tiling"].copy()).npu()
        active_mask = torch.from_numpy(reference["case0_active_mask"].copy()).npu()

    model = PagedQwenDecoderStep(
        load_checkpoint(args.model_dir),
        batch_size=BATCH_SIZE,
        physical_blocks=PHYSICAL_BLOCKS,
    ).eval().npu()
    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        token,
        position,
        sequence_length,
        block_table,
        slot_mapping,
        key_cache,
        value_cache,
        explicit_tiling,
        active_mask,
        model=model,
        export_path=str(args.output_dir),
        export_name="qwen_b4_decoder_step_attempt69c",
        dynamic=False,
        config=config,
    )
    air = args.output_dir / "qwen_b4_decoder_step_attempt69c.air"
    graph = args.output_dir / "dynamo.pbtxt"
    external_files = [
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.suffix == ""
    ]
    result = {
        "gate": "G4c Attempt 69c B=4 decoder AIR export",
        "execution_success": air.is_file() and graph.is_file(),
        "air_sha256": sha256(air) if air.is_file() else None,
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "external_file_count": len(external_files),
        "external_file_bytes": sum(path.stat().st_size for path in external_files),
        "reference_sha256": sha256(args.reference),
        "claim_boundary": (
            "AIR existence and structural audits only; native GE numerical "
            "recurrence and Device UDF remain open."
        ),
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("G4C_ATTEMPT69C_EXPORT " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["execution_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
