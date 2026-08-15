#!/usr/bin/env python3
"""Export a weight-free single-FIA graph for GraphPp packaging diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu


BATCH = 4
QUERY_HEADS = 28
KV_HEADS = 4
QUERY_TOKENS = 1
KV_TOKENS = 384
HEAD_DIM = 128


class MiniFIA(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        attention = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=QUERY_HEADS,
            num_key_value_heads=KV_HEADS,
            input_layout="BNSD",
            atten_mask=mask,
            scale=HEAD_DIM**-0.5,
            sparse_mode=0,
        )[0]
        return attention.transpose(1, 2).reshape(
            BATCH, QUERY_TOKENS, QUERY_HEADS * HEAD_DIM
        )


def inputs() -> tuple[torch.Tensor, ...]:
    query = torch.ones(
        (BATCH, QUERY_HEADS, QUERY_TOKENS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="npu",
    )
    key = torch.ones(
        (BATCH, KV_HEADS, KV_TOKENS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="npu",
    )
    value = torch.ones_like(key)
    mask = torch.zeros(
        (BATCH, 1, QUERY_TOKENS, KV_TOKENS),
        dtype=torch.bool,
        device="npu",
    )
    return query, key, value, mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    torch.npu.set_device(0)
    model = MiniFIA().eval().npu()
    sample_inputs = inputs()

    from torchair.configs.compiler_config import CompilerConfig
    from torchair.npu_export import dynamo_export

    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        *sample_inputs,
        model=model,
        export_path=str(args.output_dir),
        export_name="mini_fia",
        dynamic=False,
        config=config,
    )
    air = args.output_dir / "mini_fia.air"
    graph = args.output_dir / "dynamo.pbtxt"
    result = {
        "gate": "P3-MINI-FIA-EXPORT",
        "pass": air.is_file() and graph.is_file(),
        "eager_exact": None,
        "air_bytes": air.stat().st_size if air.is_file() else 0,
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
        "external_file_count": len(
            [path for path in args.output_dir.iterdir() if path.is_file() and not path.suffix]
        ),
        "claim_boundary": (
            "Weight-free single FusedInferAttentionScore export only; execution "
            "is checked in a separate Graph process, with no P3 Owner claim."
        ),
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
