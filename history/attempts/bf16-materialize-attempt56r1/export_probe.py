#!/usr/bin/env python3
import argparse
import hashlib
import json
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


SHAPE = (1, 1, 18944)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


enable_custom_op()
_lib = torch.library.Library("g4a_materialize", "DEF")
_lib.define("materialize(Tensor x) -> Tensor")


def _materialize_npu(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def _materialize_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x, device="meta")


_lib.impl("materialize", _materialize_npu, "PrivateUse1")
_lib.impl("materialize", _materialize_meta, "Meta")


@register_fx_node_ge_converter(torch.ops.g4a_materialize.materialize.default)
def convert_materialize(x: Tensor, *, meta_outputs: TensorSpec = None):
    return torchair.ge.custom_op("Bf16Materialize", inputs={"x": x}, outputs=["y"])


class MaterializeProbe(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.g4a_materialize.materialize(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(5601)
    source = torch.randn(SHAPE, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    input_path = args.output_dir / "input.bin"
    source.view(torch.uint16).numpy().astype("<u2", copy=False).tofile(input_path)

    torch.npu.set_device(0)
    sample = source.npu()
    model = MaterializeProbe().eval().npu()
    with torch.no_grad():
        output = model(sample)
    torch.npu.synchronize()
    eager_exact = bool(
        np.array_equal(output.view(torch.uint16).cpu().numpy(), source.view(torch.uint16).numpy())
    )
    if not eager_exact:
        raise RuntimeError("eager materialize is not bitwise identity")

    config = CompilerConfig()
    config.mode = "max-autotune"
    dynamo_export(
        sample,
        model=model,
        export_path=str(args.output_dir),
        export_name="bf16_materialize_probe",
        dynamic=False,
        config=config,
    )
    air = args.output_dir / "bf16_materialize_probe.air"
    graph = args.output_dir / "dynamo.pbtxt"
    result = {
        "execution_success": eager_exact and air.is_file() and graph.is_file(),
        "shape": list(SHAPE),
        "elements": int(np.prod(SHAPE)),
        "eager_elementwise_exact": eager_exact,
        "input_sha256": sha256(input_path),
        "air_sha256": sha256(air) if air.is_file() else None,
        "graph_sha256": sha256(graph) if graph.is_file() else None,
    }
    (args.output_dir / "export-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("BF16_MATERIALIZE_EXPORT " + json.dumps(result), flush=True)
    if not result["execution_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
