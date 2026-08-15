import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_fixed_k_graph_config_matches_lowered_ge_input_abi():
    config = json.loads(
        (ROOT / "config" / "graph_config_epoch_k6.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["inputs_tensor_desc"] == [
        {"data_type": "DT_INT32", "shape": [1]},
        {"data_type": "DT_INT32", "shape": [4]},
        {"data_type": "DT_INT64", "shape": [4, 1]},
        {"data_type": "DT_INT64", "shape": [4]},
        {"data_type": "DT_INT32", "shape": [4, 1]},
        {"data_type": "DT_BFLOAT16", "shape": [28, 4, 128, 4, 128]},
        {"data_type": "DT_INT32", "shape": [4]},
        {"data_type": "DT_BFLOAT16", "shape": [28, 4, 128, 4, 128]},
        {"data_type": "DT_UINT8", "shape": [72]},
        {"data_type": "DT_INT32", "shape": [4, 1]},
        {"data_type": "DT_INT64", "shape": [4]},
    ]


def test_fixed_k_exporter_exposes_the_matching_input_order():
    source = (
        ROOT / "experiments" / "m4b_epoch_air" / "export_fixed_k_epoch_air.py"
    ).read_text(encoding="utf-8")

    assert "block_table: torch.Tensor," in source
    assert "slot_mapping: torch.Tensor," in source
    assert "key_cache: torch.Tensor," in source
    assert "eos_token_ids: torch.Tensor," in source
    assert source.index("block_table: torch.Tensor,") < source.index(
        "slot_mapping: torch.Tensor,"
    ) < source.index("key_cache: torch.Tensor,")
    assert "K6_PHYSICAL_BLOCKS = BATCH_SIZE" in source
    assert "[[0], [1], [2], [3]]" in source
    assert "[0, BLOCK_SIZE, 2 * BLOCK_SIZE, 3 * BLOCK_SIZE]" in source


def test_fixed_k_exporter_keeps_step_limit_comparison_int32():
    source = (
        ROOT / "experiments" / "m4b_epoch_air" / "export_fixed_k_epoch_air.py"
    ).read_text(encoding="utf-8")

    assert source.count("torch.full_like(step_limit.reshape(1), step)") == 2
    assert "step_limit.reshape(1) > step_index" in source


def test_fixed_k_exporter_uses_static_per_row_kv_updates():
    source = (
        ROOT / "experiments" / "m4b_epoch_air" / "export_fixed_k_epoch_air.py"
    ).read_text(encoding="utf-8")

    assert "class FixedLayoutPagedQwenDecoderStep(PagedQwenDecoderStep)" in source
    assert '"fixed_physical_slots"' in source
    assert "flat = layer_cache.reshape(" in source
    assert "for batch_index in range(BATCH_SIZE):" in source
    assert "updated_flat = torch.where(update_mask, replacement, updated_flat)" in source
    assert "torch.scatter(" not in source
    assert "dense = updated_flat.reshape(" in source


def test_fixed_k_controller_uses_lowered_ge_input_order():
    source = (ROOT / "controller" / "g4c_b4_resident_epoch.cpp").read_text(
        encoding="utf-8"
    )

    assert "model_inputs = {step_limit, current_active, current_token," in source
    assert "current_slot, current_value, tiling_input," in source
    assert "block_table_input, eos_token_ids};" in source


def test_direct_device_import_uses_metadata_fingerprint_not_payload_scan():
    controller = (ROOT / "controller" / "g4c_b4_resident_epoch.cpp").read_text(
        encoding="utf-8"
    )
    bridge = (ROOT / "native" / "resident_epoch_bridge.cpp").read_text(
        encoding="utf-8"
    )
    transfer = (ROOT / "native" / "resident_device_transfer.cpp").read_text(
        encoding="utf-8"
    )

    assert "kDirectDeviceImportGraphFlag = 0x200" in controller
    assert "kDirectDeviceImportGraphFlag = 0x200" in bridge
    assert "IpcMetadataFingerprint" in bridge
    assert "direct_device_import ? static_cast<int32_t>(expected_import_checksum)" in bridge
    assert "if (direct_device_import) {\n        import_checksum = static_cast<uint32_t>(sampling_mode);" in controller
    assert "if (!direct_device_import &&\n            (!ClearCacheRow(resident_key_, request, layout) ||" in controller
    assert 'ResolveAclSymbol("aclrtMemset"' not in transfer
    assert "aclrtMemset" not in transfer


def test_fixed_k_graph_stops_each_row_after_its_eos_token():
    source = (
        ROOT / "experiments" / "m4b_epoch_air" / "export_fixed_k_epoch_air.py"
    ).read_text(encoding="utf-8")
    controller = (ROOT / "controller" / "g4c_b4_resident_epoch.cpp").read_text(
        encoding="utf-8"
    )

    assert "(generated_flat != eos_token_ids).to(torch.int32)" in source
    assert "current_position_values[request] + executed[request]" in controller


def test_fixed_k_controller_and_bridge_use_the_compact_kv_layout():
    controller = (ROOT / "controller" / "g4c_b4_resident_epoch.cpp").read_text(
        encoding="utf-8"
    )
    bridge = (ROOT / "native" / "resident_epoch_bridge.cpp").read_text(
        encoding="utf-8"
    )

    assert "kFixedEpochPhysicalBlocks = 4" in controller
    assert "kFixedEpochBlocksPerRequest = 1" in controller
    assert "kFixedEpochCacheLayout" in controller
    assert "kFixedEpochBlocksPerRequest = 1" in bridge
    assert "{4, blocks_per_request}" in bridge


def test_fixed_k_npu_smoke_covers_recurrence_continuity_and_per_row_eos():
    source = (
        ROOT / "experiments" / "m4b_epoch_air" / "smoke_fixed_k_epoch.py"
    ).read_text(encoding="utf-8")

    assert "run_one_step_reference" in source
    assert "k6-continuation" in source
    assert "k6-eos" in source
    assert "EOS terminal KV state differs" in source
    assert "active_rows=(0, 2)" in source
    assert "for row in (0, 2)" in source
    assert "sidecar_environment" in source
    assert 'getattr(args, name).resolve(strict=True)' in source
    assert source.index('getattr(args, name).resolve(strict=True)') < source.index(
        "os.chdir(args.output.parent.resolve())"
    )
