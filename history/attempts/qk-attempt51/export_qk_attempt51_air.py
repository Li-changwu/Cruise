#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from torchair.configs.compiler_config import CompilerConfig
from torchair.npu_export import dynamo_export

ATTEMPT50_SRC = Path(__file__).resolve().parent.parent / "attempt50-src"
sys.path.insert(0, str(ATTEMPT50_SRC))
from export_qk_attempt50_air import (  # noqa: E402
    HEAD_DIM,
    TILING_WORDS,
    bits_to_npu,
    prepare_operands,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaterializedBf16Probe(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor, explicit_tiling: torch.Tensor):
        raw = torch.ops.g4a_qk.exact_qk(a, b, explicit_tiling)
        scale = math.sqrt(HEAD_DIM)
        legacy_bf16 = raw / scale
        fp32_div_bf16 = (raw.float() / scale).to(torch.bfloat16).unsqueeze(2)
        fp32_mul_bf16 = (raw.float() * (1.0 / scale)).to(torch.bfloat16).unsqueeze(2)
        return raw, legacy_bf16.unsqueeze(2), fp32_div_bf16, fp32_mul_bf16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt8-output", type=Path, required=True)
    parser.add_argument("--eager-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-input-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.native_input_dir.mkdir(parents=True, exist_ok=True)

    attempt8 = np.load(args.attempt8_output)
    tiling = np.asarray(TILING_WORDS, dtype="<u4").view(np.uint8).copy()
    tiling.tofile(args.native_input_dir / "tiling.bin")
    operands = []
    for step in range(1, 5):
        a, b = prepare_operands(
            attempt8[f"step{step}_q_rope_bf16"],
            attempt8[f"step{step}_updated_key_bf16"],
        )
        a.tofile(args.native_input_dir / f"step{step}_a.bin")
        b.tofile(args.native_input_dir / f"step{step}_b.bin")
        operands.append((a, b))

    torch.npu.set_device(0)
    model = MaterializedBf16Probe().eval().npu()
    sample_a = bits_to_npu(operands[0][0])
    sample_b = bits_to_npu(operands[0][1])
    sample_tiling = torch.from_numpy(tiling).npu()
    with torch.no_grad():
        outputs = model(sample_a, sample_b, sample_tiling)
    torch.npu.synchronize()
    if len(outputs) != 4 or any(output.dtype != torch.bfloat16 for output in outputs):
        raise RuntimeError("expected four materialized BF16 outputs")

    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        sample_a,
        sample_b,
        sample_tiling,
        model=model,
        export_path=str(args.output_dir),
        export_name="qk_bf16_scaling_probe",
        dynamic=False,
        config=config,
    )
    air = args.output_dir / "qk_bf16_scaling_probe.air"
    graph = args.output_dir / "dynamo.pbtxt"
    result = {
        "execution_success": air.is_file() and graph.is_file(),
        "attempt8_output_sha256": sha256(args.attempt8_output),
        "eager_reference_sha256": sha256(args.eager_reference),
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("G4A_QK_ATTEMPT51_EXPORT " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["execution_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

