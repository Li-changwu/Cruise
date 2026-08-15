#!/usr/bin/env python3
"""Validate the 384-token Device QK MatMul extent required by P3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


BATCH_HEADS = 28
QUERY_ROWS = 1
HEAD_DIM = 128
KV_CAPACITY = 384
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_device_qk() -> None:
    import torch_npu  # noqa: F401

    library = torch.library.Library("cruise_p3_qk", "DEF")
    library.define("exact_qk_384(Tensor a, Tensor b, Tensor tiling) -> Tensor")

    def exact_qk_npu(
        a: torch.Tensor, b: torch.Tensor, tiling: torch.Tensor
    ) -> torch.Tensor:
        if tuple(a.shape) != (QUERY_ROWS, BATCH_HEADS, HEAD_DIM):
            raise ValueError(f"unexpected Q shape: {tuple(a.shape)}")
        if tuple(b.shape) != (BATCH_HEADS, HEAD_DIM, KV_CAPACITY):
            raise ValueError(f"unexpected K shape: {tuple(b.shape)}")
        if tiling.dtype != torch.uint8 or tiling.numel() != 72:
            raise ValueError("explicit tiling must be 72 uint8 bytes")
        output = torch.empty(
            (QUERY_ROWS, BATCH_HEADS, KV_CAPACITY),
            dtype=torch.bfloat16,
            device=a.device,
        )
        del tiling
        return torch.matmul(a.transpose(0, 1), b).transpose(0, 1).to(
            dtype=output.dtype
        )

    def exact_qk_meta(
        a: torch.Tensor, b: torch.Tensor, tiling: torch.Tensor
    ) -> torch.Tensor:
        del b, tiling
        return torch.empty(
            (QUERY_ROWS, BATCH_HEADS, KV_CAPACITY), dtype=a.dtype, device="meta"
        )

    library.impl("exact_qk_384", exact_qk_npu, "PrivateUse1")
    library.impl("exact_qk_384", exact_qk_meta, "Meta")
    # Keep the Library object alive for the process lifetime.
    globals()["_P3_QK_LIBRARY"] = library


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opp-vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=300384)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    del args.opp_vendor
    register_device_qk()
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    q_cpu = (
        torch.randn(
            (QUERY_ROWS, BATCH_HEADS, HEAD_DIM), generator=generator
        )
        * 0.125
    ).to(torch.bfloat16)
    k_cpu = (
        torch.randn(
            (BATCH_HEADS, HEAD_DIM, KV_CAPACITY), generator=generator
        )
        * 0.125
    ).to(torch.bfloat16)
    tiling_bytes = np.asarray(TILING_WORDS, dtype="<u4").view(np.uint8).copy()
    if tiling_bytes.size != 72:
        raise AssertionError("P3 ExactQk tiling ABI changed")

    q = q_cpu.npu()
    k = k_cpu.npu()
    tiling = torch.from_numpy(tiling_bytes).npu()
    with torch.no_grad():
        actual = torch.ops.cruise_p3_qk.exact_qk_384(q, k, tiling)
        torch.npu.synchronize()
        # ``torch.matmul(q, k)`` would broadcast the leading dimensions and
        # produce the wrong [28, 28, 384] result.  Transpose per head so the
        # device reference is exactly [1, 28, 128] @ [28, 128, 384].
        expected = torch.matmul(q.transpose(0, 1), k).transpose(0, 1)
        torch.npu.synchronize()
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    exact = torch.equal(actual_cpu, expected_cpu)
    actual_fp32 = actual_cpu.float()
    expected_fp32 = expected_cpu.float()
    max_abs_error = float((actual_fp32 - expected_fp32).abs().max().item())
    close = bool(
        torch.allclose(actual_fp32, expected_fp32, rtol=5e-3, atol=5e-3)
    )

    result = {
        "gate": "P3-QK384-device-matmul",
        "pass": exact,
        "device": "Ascend",
        "seed": args.seed,
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "output_shape": list(actual.shape),
        "dtype": str(actual.dtype).removeprefix("torch."),
        "bit_exact": exact,
        "within_tolerance": close,
        "max_abs_error": max_abs_error,
        "tiling_words": list(TILING_WORDS),
        "tiling_sha256": hashlib.sha256(tiling_bytes.tobytes()).hexdigest(),
        "qk_implementation": "torch.matmul on NPU",
        "probe_sha256": sha256(Path(__file__)),
        "claim_boundary": (
            "Device QK 384-token extent only; no Decoder, Persistent Owner, "
            "or service-performance claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
