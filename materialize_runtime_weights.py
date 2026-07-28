#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_CONFIG_SHA256 = (
    "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c"
)
MODEL_INDEX_SHA256 = (
    "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028"
)
NUM_LAYERS = 28
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_SIZE = 128
VOCAB_SIZE = 152064
ROPE_THETA = 1_000_000.0
PHYSICAL_BLOCKS = 8
BLOCK_SIZE = 128
LOGICAL_CAPACITY = 8
EXPECTED_CHECKPOINT_TENSORS = 339
EXPECTED_EXTERNAL_FILES = 342
EXPECTED_EXTERNAL_BYTES = 15_231_237_408


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layer_key_names(index: int) -> dict[str, str]:
    prefix = f"model.layers.{index}"
    return {
        f"{prefix}.input_layernorm.weight": "input_norm",
        f"{prefix}.post_attention_layernorm.weight": "post_norm",
        f"{prefix}.self_attn.q_proj.weight": "q_weight",
        f"{prefix}.self_attn.q_proj.bias": "q_bias",
        f"{prefix}.self_attn.k_proj.weight": "k_weight",
        f"{prefix}.self_attn.k_proj.bias": "k_bias",
        f"{prefix}.self_attn.v_proj.weight": "v_weight",
        f"{prefix}.self_attn.v_proj.bias": "v_bias",
        f"{prefix}.self_attn.o_proj.weight": "o_weight",
        f"{prefix}.mlp.gate_proj.weight": "gate_weight",
        f"{prefix}.mlp.up_proj.weight": "up_weight",
        f"{prefix}.mlp.down_proj.weight": "down_weight",
    }


def checkpoint_destinations() -> dict[str, str]:
    destinations = {
        "model.embed_tokens.weight": "embedding",
        "model.norm.weight": "final_norm",
        "lm_head.weight": "lm_head",
    }
    for layer in range(NUM_LAYERS):
        for key, local_name in layer_key_names(layer).items():
            destinations[key] = f"layers_{layer}_{local_name}"
    return destinations


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "model.embed_tokens.weight": (VOCAB_SIZE, HIDDEN_SIZE),
        "model.norm.weight": (HIDDEN_SIZE,),
        "lm_head.weight": (VOCAB_SIZE, HIDDEN_SIZE),
    }
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (HIDDEN_SIZE,),
                f"{prefix}.post_attention_layernorm.weight": (HIDDEN_SIZE,),
                f"{prefix}.self_attn.q_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
                f"{prefix}.self_attn.q_proj.bias": (HIDDEN_SIZE,),
                f"{prefix}.self_attn.k_proj.weight": (
                    NUM_KV_HEADS * HEAD_SIZE,
                    HIDDEN_SIZE,
                ),
                f"{prefix}.self_attn.k_proj.bias": (
                    NUM_KV_HEADS * HEAD_SIZE,
                ),
                f"{prefix}.self_attn.v_proj.weight": (
                    NUM_KV_HEADS * HEAD_SIZE,
                    HIDDEN_SIZE,
                ),
                f"{prefix}.self_attn.v_proj.bias": (
                    NUM_KV_HEADS * HEAD_SIZE,
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    HIDDEN_SIZE,
                    HIDDEN_SIZE,
                ),
                f"{prefix}.mlp.gate_proj.weight": (
                    INTERMEDIATE_SIZE,
                    HIDDEN_SIZE,
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    INTERMEDIATE_SIZE,
                    HIDDEN_SIZE,
                ),
                f"{prefix}.mlp.down_proj.weight": (
                    HIDDEN_SIZE,
                    INTERMEDIATE_SIZE,
                ),
            }
        )
    return shapes


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().cpu().contiguous()
    byte_view = contiguous.view(torch.uint8).numpy()
    with path.open("xb") as stream:
        byte_view.tofile(stream)
    expected_bytes = contiguous.numel() * contiguous.element_size()
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"short tensor write: {path}")


def validate_model_identity(model_dir: Path) -> tuple[Path, Path]:
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if sha256(config_path) != MODEL_CONFIG_SHA256:
        raise RuntimeError("model config hash does not match the frozen revision")
    if sha256(index_path) != MODEL_INDEX_SHA256:
        raise RuntimeError("model index hash does not match the frozen revision")
    return config_path, index_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if not str(output_dir).startswith("/dev/shm/"):
        raise RuntimeError("runtime weights must be materialized under /dev/shm")
    output_dir.mkdir(parents=True, exist_ok=False)

    config_path, index_path = validate_model_identity(model_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    destinations = checkpoint_destinations()
    shapes = expected_shapes()
    if len(destinations) != EXPECTED_CHECKPOINT_TENSORS:
        raise RuntimeError("internal checkpoint mapping count changed")
    observed_keys = set(index["weight_map"])
    if observed_keys != set(destinations):
        raise RuntimeError("checkpoint keys do not match the frozen decoder")
    expected_config = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "num_hidden_layers": NUM_LAYERS,
        "vocab_size": VOCAB_SIZE,
        "rope_theta": ROPE_THETA,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("model configuration does not match the frozen decoder")

    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in index["weight_map"].items():
        keys_by_shard[shard].append(key)

    files: list[dict[str, object]] = []
    for shard in sorted(keys_by_shard):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for key in sorted(keys_by_shard[shard]):
                tensor = handle.get_tensor(key)
                if tensor.dtype != torch.bfloat16:
                    raise RuntimeError(f"unexpected dtype for {key}: {tensor.dtype}")
                if tuple(tensor.shape) != shapes[key]:
                    raise RuntimeError(f"unexpected shape for {key}: {tensor.shape}")
                destination = output_dir / destinations[key]
                write_tensor(destination, tensor)
                files.append(
                    {
                        "name": destination.name,
                        "source": key,
                        "shape": list(tensor.shape),
                        "dtype": "bfloat16",
                        "bytes": destination.stat().st_size,
                        "sha256": sha256(destination),
                    }
                )

    derived = {
        "inv_freq": 1.0
        / (
            ROPE_THETA
            ** (
                torch.arange(0, HEAD_SIZE, 2, dtype=torch.float32)
                / HEAD_SIZE
            )
        ),
        "physical_slots": torch.arange(
            PHYSICAL_BLOCKS * BLOCK_SIZE, dtype=torch.int32
        ),
        "logical_offsets": torch.arange(LOGICAL_CAPACITY, dtype=torch.int32),
    }
    for name, tensor in derived.items():
        destination = output_dir / name
        write_tensor(destination, tensor)
        files.append(
            {
                "name": name,
                "source": "derived",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    files.sort(key=lambda item: str(item["name"]))
    total_bytes = sum(int(item["bytes"]) for item in files)
    if len(files) != EXPECTED_EXTERNAL_FILES:
        raise RuntimeError(f"unexpected external file count: {len(files)}")
    if total_bytes != EXPECTED_EXTERNAL_BYTES:
        raise RuntimeError(f"unexpected external byte count: {total_bytes}")
    if len({str(item["name"]) for item in files}) != len(files):
        raise RuntimeError("duplicate external weight destination")

    manifest = {
        "gate": "Attempt 71 runtime external-weight materialization",
        "pass": True,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_index_sha256": MODEL_INDEX_SHA256,
        "checkpoint_tensors": EXPECTED_CHECKPOINT_TENSORS,
        "derived_tensors": len(derived),
        "external_files": len(files),
        "external_bytes": total_bytes,
        "files": files,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "ATTEMPT71_RUNTIME_WEIGHTS "
        + json.dumps(
            {key: value for key, value in manifest.items() if key != "files"},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
