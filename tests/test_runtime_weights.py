from pathlib import Path

import pytest

import materialize_runtime_weights as runtime_weights
from materialize_runtime_weights import (
    ASSET_STORE_MARKER,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_CHECKPOINT_TENSORS,
    checkpoint_destinations,
    expected_shapes,
    prepare_asset_store,
    resolve_output_location,
)


def test_runtime_weight_mapping_covers_frozen_decoder_buffers():
    destinations = checkpoint_destinations()
    shapes = expected_shapes()

    assert len(destinations) == EXPECTED_CHECKPOINT_TENSORS == 339
    assert set(destinations) == set(shapes)
    assert destinations["model.embed_tokens.weight"] == "embedding"
    assert destinations["model.layers.0.self_attn.q_proj.weight"] == (
        "layers_0_q_weight"
    )
    assert destinations["model.layers.27.mlp.down_proj.weight"] == (
        "layers_27_down_weight"
    )
    assert destinations["model.norm.weight"] == "final_norm"
    assert destinations["lm_head.weight"] == "lm_head"
    assert len(set(destinations.values())) == EXPECTED_CHECKPOINT_TENSORS


def test_frozen_model_identity_is_path_independent(tmp_path, monkeypatch):
    model_dir = tmp_path / "shared-model-with-arbitrary-name"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config_path.write_text('{"model_type": "qwen2"}\n', encoding="utf-8")
    index_path.write_text('{"weight_map": {}}\n', encoding="utf-8")
    monkeypatch.setattr(
        runtime_weights, "MODEL_CONFIG_SHA256", runtime_weights.sha256(config_path)
    )
    monkeypatch.setattr(
        runtime_weights, "MODEL_INDEX_SHA256", runtime_weights.sha256(index_path)
    )

    assert runtime_weights.validate_model_identity(model_dir) == (
        config_path,
        index_path,
    )

    config_path.write_text('{"model_type": "other"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="model config hash"):
        runtime_weights.validate_model_identity(model_dir)


def test_persistent_output_is_bound_to_manifest_digest(tmp_path):
    # The runner intentionally places pytest's TMPDIR under /dev/shm. Use a
    # non-materialized persistent path so this path-resolution test is
    # independent of the temporary-directory policy.
    asset_root = Path("/var/tmp") / tmp_path.name / "cruise-assets"
    expected = asset_root / "runtime-weights" / EXPECTED_MANIFEST_SHA256

    output, root = resolve_output_location(expected, asset_root)

    assert output == expected.resolve()
    assert root == asset_root.resolve()
    with pytest.raises(RuntimeError, match="content-addressed path"):
        resolve_output_location(asset_root / "runtime-weights" / "latest", asset_root)


def test_asset_store_refuses_nonempty_unmarked_directory(tmp_path):
    asset_root = tmp_path / "cruise-assets"
    asset_root.mkdir()
    (asset_root / "unknown").write_text("keep", encoding="ascii")

    with pytest.raises(RuntimeError, match="non-empty unmarked"):
        prepare_asset_store(asset_root)

    assert (asset_root / "unknown").is_file()
    assert not (asset_root / ASSET_STORE_MARKER).exists()


def test_asset_store_marker_is_idempotent(tmp_path):
    asset_root = tmp_path / "cruise-assets"

    prepare_asset_store(asset_root)
    prepare_asset_store(asset_root)

    assert (asset_root / ASSET_STORE_MARKER).is_file()
