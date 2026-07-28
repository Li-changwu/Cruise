#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.npu_export import dynamo_export
from vllm_ascend.utils import enable_custom_op


NUM_HEADS = 28
NUM_KV_HEADS = 4
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
MAX_KV = 8
HEAD_DIM = 128
TILING_WORDS = (28, 1, 128, 8, 16, 512, 16, 1, 1, 1, 28, 5, 2336, 24, 0, 0, 0, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_operands(q_bits: np.ndarray, key_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if q_bits.dtype != np.uint16 or key_bits.dtype != np.uint16:
        raise RuntimeError("expected raw uint16 BF16 operands")
    a = np.ascontiguousarray(q_bits.squeeze(2))
    expanded = np.repeat(key_bits, NUM_KV_GROUPS, axis=1)
    b = np.ascontiguousarray(expanded.squeeze(0).transpose(0, 2, 1))
    if a.shape != (1, NUM_HEADS, HEAD_DIM) or b.shape != (NUM_HEADS, HEAD_DIM, MAX_KV):
        raise RuntimeError(f"unexpected operand shapes a={a.shape} b={b.shape}")
    return a, b


def bits_to_npu(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(bits)).view(torch.bfloat16).npu()


enable_custom_op()
_lib = torch.library.Library("g4a_qk", "DEF")
_lib.define("exact_qk(Tensor a, Tensor b, Tensor explicit_tiling) -> Tensor")


def _exact_qk_npu(a: torch.Tensor, b: torch.Tensor, explicit_tiling: torch.Tensor) -> torch.Tensor:
    if explicit_tiling.dtype != torch.uint8 or explicit_tiling.numel() != 72:
        raise RuntimeError("explicit tiling must be a 72-byte uint8 tensor")
    output = torch.empty((1, NUM_HEADS, MAX_KV), dtype=torch.bfloat16, device=a.device)
    torch.ops._C_ascend.batch_matmul_transpose(a, b, output)
    return output


def _exact_qk_meta(a: torch.Tensor, b: torch.Tensor, explicit_tiling: torch.Tensor) -> torch.Tensor:
    return torch.empty((1, NUM_HEADS, MAX_KV), dtype=a.dtype, device="meta")


_lib.impl("exact_qk", _exact_qk_npu, "PrivateUse1")
_lib.impl("exact_qk", _exact_qk_meta, "Meta")


@register_fx_node_ge_converter(torch.ops.g4a_qk.exact_qk.default)
def convert_exact_qk(
    a: Tensor,
    b: Tensor,
    explicit_tiling: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "ExactQk",
        inputs={"a": a, "b": b, "explicit_tiling": explicit_tiling},
        outputs=["c"],
    )


class QkBoundaryProbe(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor, explicit_tiling: torch.Tensor):
        raw = torch.ops.g4a_qk.exact_qk(a, b, explicit_tiling)
        scale = math.sqrt(HEAD_DIM)
        legacy_scaled = (raw / scale).unsqueeze(2).float()
        fp32_div_bf16 = (raw.float() / scale).to(torch.bfloat16).unsqueeze(2).float()
        fp32_mul_bf16 = (raw.float() * (1.0 / scale)).to(torch.bfloat16).unsqueeze(2).float()
        return raw, legacy_scaled, fp32_div_bf16, fp32_mul_bf16


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
    model = QkBoundaryProbe().eval().npu()
    sample_a = bits_to_npu(operands[0][0])
    sample_b = bits_to_npu(operands[0][1])
    sample_tiling = torch.from_numpy(tiling).npu()
    with torch.no_grad():
        outputs = model(sample_a, sample_b, sample_tiling)
    torch.npu.synchronize()
    if len(outputs) != 4:
        raise RuntimeError("eager probe did not return four outputs")

    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        sample_a,
        sample_b,
        sample_tiling,
        model=model,
        export_path=str(args.output_dir),
        export_name="qk_boundary_probe",
        dynamic=False,
        config=config,
    )
    air = args.output_dir / "qk_boundary_probe.air"
    graph = args.output_dir / "dynamo.pbtxt"
    result = {
        "execution_success": air.is_file() and graph.is_file(),
        "attempt8_output_sha256": sha256(args.attempt8_output),
        "eager_reference_sha256": sha256(args.eager_reference),
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "tiling_sha256": sha256(args.native_input_dir / "tiling.bin"),
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("G4A_QK_ATTEMPT50_EXPORT " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["execution_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

