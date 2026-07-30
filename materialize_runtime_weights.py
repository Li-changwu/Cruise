#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
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
EXPECTED_MANIFEST_SHA256 = (
    "2ec95bf8e78cfaf091782b3c531b19b9cced35dcfab0e418c756e25abe456761"
)
ASSET_STORE_MARKER = ".cruise-asset-store-v1"
ASSET_STORE_MARKER_CONTENT = "Cruise content-addressed asset store v1\n"


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


def resolve_output_location(
    output_dir: Path, persistent_asset_root: Path | None
) -> tuple[Path, Path | None]:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to("/dev/shm")
    except ValueError:
        pass
    else:
        if output_dir == Path("/dev/shm"):
            raise RuntimeError("runtime-weight output must not be /dev/shm itself")
        return output_dir, None

    if persistent_asset_root is None:
        raise RuntimeError(
            "runtime weights outside /dev/shm require --persistent-asset-root"
        )
    asset_root = persistent_asset_root.resolve()
    if asset_root == Path(asset_root.anchor):
        raise RuntimeError("persistent asset root must be a dedicated child directory")
    expected = asset_root / "runtime-weights" / EXPECTED_MANIFEST_SHA256
    if output_dir != expected:
        raise RuntimeError(
            "persistent runtime-weight output must equal the content-addressed path "
            f"{expected}"
        )
    return output_dir, asset_root


def prepare_asset_store(asset_root: Path) -> None:
    marker = asset_root / ASSET_STORE_MARKER
    if asset_root.exists():
        if not asset_root.is_dir():
            raise RuntimeError(f"persistent asset root is not a directory: {asset_root}")
        if marker.exists():
            if marker.read_text(encoding="ascii") != ASSET_STORE_MARKER_CONTENT:
                raise RuntimeError(f"persistent asset marker is invalid: {marker}")
            return
        if any(asset_root.iterdir()):
            raise RuntimeError(
                f"refusing to adopt non-empty unmarked asset root: {asset_root}"
            )
    else:
        asset_root.mkdir(parents=True, mode=0o700)
    marker.write_text(ASSET_STORE_MARKER_CONTENT, encoding="ascii")


def validate_existing_bundle(output_dir: Path, manifest_path: Path) -> None:
    if not output_dir.is_dir() or not manifest_path.is_file():
        raise RuntimeError(
            "content-addressed runtime-weight bundle is incomplete: "
            f"weights={output_dir}, manifest={manifest_path}"
        )
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("existing runtime-weight manifest hash is incompatible")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != EXPECTED_EXTERNAL_FILES:
        raise RuntimeError("existing runtime-weight manifest file list is incomplete")
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    if len(files) != EXPECTED_EXTERNAL_FILES:
        raise RuntimeError("existing runtime-weight file count is incompatible")
    if sum(path.stat().st_size for path in files) != EXPECTED_EXTERNAL_BYTES:
        raise RuntimeError("existing runtime-weight byte count is incompatible")
    by_name = {
        record.get("name"): record for record in records if isinstance(record, dict)
    }
    if set(by_name) != {path.name for path in files}:
        raise RuntimeError("existing runtime-weight names are incompatible")
    for path in files:
        record = by_name[path.name]
        if path.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"existing runtime-weight size mismatch: {path.name}")
        if sha256(path) != record.get("sha256"):
            raise RuntimeError(f"existing runtime-weight hash mismatch: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--persistent-asset-root", type=Path)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve(strict=True)
    target_dir, asset_root = resolve_output_location(
        args.output_dir, args.persistent_asset_root
    )
    manifest_path = args.manifest.resolve()
    if asset_root is not None:
        prepare_asset_store(asset_root)
        expected_manifest = (
            asset_root / "manifests" / f"{EXPECTED_MANIFEST_SHA256}.json"
        )
        if manifest_path != expected_manifest:
            raise RuntimeError(
                "persistent runtime-weight manifest must equal the "
                f"content-addressed path {expected_manifest}"
            )
        if target_dir.exists():
            validate_existing_bundle(target_dir, manifest_path)
            print(
                "CRUISE_RUNTIME_WEIGHTS "
                + json.dumps(
                    {
                        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                        "output_dir": str(target_dir),
                        "reused": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = target_dir.with_name(
            f".{target_dir.name}.staging-{os.getpid()}"
        )
        temporary_manifest = manifest_path.with_name(
            f".{manifest_path.name}.staging-{os.getpid()}"
        )
    else:
        output_dir = target_dir
        temporary_manifest = manifest_path

    cleanup_enabled = True

    def cleanup_partial_output() -> None:
        if not cleanup_enabled:
            return
        if output_dir != target_dir and output_dir.exists():
            shutil.rmtree(output_dir)
        if temporary_manifest != manifest_path:
            temporary_manifest.unlink(missing_ok=True)

    atexit.register(cleanup_partial_output)
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
    temporary_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    observed_manifest_sha256 = sha256(temporary_manifest)
    if observed_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "runtime-weight manifest changed: "
            f"expected {EXPECTED_MANIFEST_SHA256}, observed "
            f"{observed_manifest_sha256}"
        )
    if asset_root is not None:
        output_dir.rename(target_dir)
        temporary_manifest.replace(manifest_path)
    cleanup_enabled = False
    print(
        "CRUISE_RUNTIME_WEIGHTS "
        + json.dumps(
            {
                **{key: value for key, value in manifest.items() if key != "files"},
                "manifest_sha256": observed_manifest_sha256,
                "output_dir": str(target_dir),
                "reused": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
