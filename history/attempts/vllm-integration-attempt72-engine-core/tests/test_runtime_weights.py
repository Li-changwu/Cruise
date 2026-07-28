from materialize_runtime_weights import (
    EXPECTED_CHECKPOINT_TENSORS,
    checkpoint_destinations,
    expected_shapes,
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
