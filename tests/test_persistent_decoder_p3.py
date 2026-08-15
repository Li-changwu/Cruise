import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

from experiments.persistent_decoder_p3.compare_p3_gate import compare
from experiments.persistent_decoder_p3.promote_ge_external_weights import source_records
from experiments.persistent_decoder_p3.promote_p3_air_equal_prefix import (
    file_constant_paths,
    replace_prefix,
    validate_published,
)


ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "experiments" / "persistent_decoder_p3"


def _assignment(module: ast.Module, name: str):
    for node in module.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment: {name}")


def test_p3_protocol_keeps_decoder_claim_separate_from_performance():
    protocol = (P3 / "protocol.md").read_text(encoding="utf-8")

    assert "not a service-performance result" in protocol
    assert "384 tokens total" in protocol
    assert "may not send a token-step command" in protocol
    assert "Device Decode coverage must be 100%" in protocol
    assert "three independent Owner starts" in protocol


def test_p3_qk_tiling_covers_the_384_token_extent():
    source = (P3 / "qk_capacity_probe.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    words = _assignment(module, "TILING_WORDS")

    assert len(words) == 18
    assert words[0:4] == (28, 1, 128, 384)
    assert words[6] == 128
    assert words[7:11] == (1, 1, 3, 84)
    assert words[13] == 24
    assert "torch.matmul(a.transpose(0, 1), b).transpose(0, 1)" in source
    assert '"pass": exact' in source
    assert "torch.matmul(q, k)" in source


def test_p3_decoder_air_uses_the_actual_eight_input_abi():
    source = (P3 / "export_p3_decoder.py").read_text(encoding="utf-8")

    assert "explicit_tiling: torch.Tensor" not in source
    assert '["DT_UINT8", [72]]' not in source
    assert '"qk_probe_tiling_words"' in source
    assert '"external_file_count"' in source
    assert '"air_sha256"' in source


def test_p3_decoder_matches_vllm_ascend_operator_boundaries():
    source = (P3 / "export_p3_decoder.py").read_text(encoding="utf-8")

    assert 'layer.register_buffer("qkv_weight"' in source
    assert 'layer.register_buffer("gate_up_weight"' in source
    assert "torch_npu.npu_rms_norm" in source
    assert "torch_npu.npu_add_rms_norm" in source
    assert "torch_npu.npu_rotary_mul" in source
    assert "torch_npu.npu_fused_infer_attention_score" in source
    assert 'input_layout="BNSD"' in source
    assert "atten_mask=attention_mask" in source
    assert "torch.softmax" not in source
    assert "STOCK_VLLM_TOKEN_ORACLE" in source
    assert "eager_stock_semantics_check(model)" in source
    assert '"eager_stock_vllm_semantics_exact"' in source


def test_p3_export_inspector_requires_actual_graph_signatures():
    source = (P3 / "inspect_p3_export.py").read_text(encoding="utf-8")

    assert 'CACHE_SIGNATURE = ("DT_BF16", (28, 12, 128, 4, 128))' in source
    assert '("DT_INT32", (4, 3)): "block_table"' in source
    assert '"Data": 8' in source
    assert '"ArgMaxV2": 1' in source
    assert '"FileConstant": 202' in source
    assert '"FusedInferAttentionScore": 28' in source
    assert 'FORBIDDEN_OPS = ("SoftmaxV2", "BatchMatMul")' in source
    assert "file_constants_valid" in source


def test_p3_graph_externalizes_model_weights_and_owner_requires_output_dir():
    config_source = (P3 / "prepare_p3_config.py").read_text(encoding="utf-8")
    host_source = (P3 / "persistent_decoder_p3_host.cpp").read_text(encoding="utf-8")
    oracle_runner = (P3 / "run_graph_oracle_on_910b.sh").read_text(encoding="utf-8")
    promoter = (P3 / "promote_ge_external_weights.py").read_text(encoding="utf-8")

    assert '"ge.externalWeight": "1"' in config_source
    assert 'CRUISE_P3_EXTERNAL_WEIGHT_DIR' in host_source
    assert '"ge.externalWeightDir"' in host_source
    assert 'external_weight_dir == run_root + "/.ge-external-view"' in host_source
    assert "external-weight runtime path is not traversable" in oracle_runner
    assert "ensure_runtime_traversal(assets_root)" in promoter

    owner_runner = (P3 / "run_on_910b.sh").read_text(encoding="utf-8")
    for runner in (oracle_runner, owner_runner):
        assert "p3-air384-fia-37bd7557850a72b4/qwen_b4_p3_decoder_step.runtime.air" in runner
        assert "p3-ge-external-420a16406d4f8723" in runner
        assert "STORAGE_GUARD_PROJECT_AUDIT_INTERVAL_SECONDS=2" in runner
        assert '"${evidence}/source-identity.sha256"' in runner
        assert '"${external_weights}/dedup-manifest.json"' in runner
    assert "cruise-p3-oracle-${physical_npu}-$$" in oracle_runner
    assert "cruise-p3-owner-${physical_npu}-$$" in owner_runner
    assert "CRUISE_P3_MAX_SCRATCH_GIB:-2" in owner_runner


def test_p3_ge_external_view_is_mutable_but_canonical_bundle_is_unchanged(tmp_path):
    tool = P3 / "manage_ge_external_view.py"
    source = tmp_path / "assets" / "bundle"
    run_root = tmp_path / "runs" / "run-1"
    evidence = run_root / "evidence"
    source.mkdir(parents=True)
    evidence.mkdir(parents=True)
    weight = source / "weight_abc"
    weight.write_bytes(b"weight")
    meta = {
        "hash_to_weight_file": {"abc": str(weight)},
        "hash_to_weight_offset": {"abc": 0},
    }
    (source / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (source / "dedup-manifest.json").write_text(
        json.dumps(
            {
                "valid": True,
                "logical_file_count": 1,
                "identity_sha256": "identity",
            }
        ),
        encoding="utf-8",
    )
    source_meta_before = (source / "meta.json").read_bytes()
    view = run_root / ".ge-external-view"
    receipt = evidence / "ge-external-view-cleanup.json"

    subprocess.run(
        [
            "python3",
            str(tool),
            "create",
            "--source",
            str(source),
            "--run-root",
            str(run_root),
            "--view",
            str(view),
        ],
        check=True,
    )
    assert os.stat(view / "meta.json").st_mode & 0o600 == 0o600
    view_meta = json.loads((view / "meta.json").read_text())
    view_weight = Path(view_meta["hash_to_weight_file"]["abc"])
    assert view_weight.parent == view
    assert view_weight.is_symlink()
    assert view_weight.resolve() == weight
    (view / "meta.json").write_text("runtime mutation", encoding="utf-8")
    assert (source / "meta.json").read_bytes() == source_meta_before

    subprocess.run(
        [
            "python3",
            str(tool),
            "cleanup",
            "--run-root",
            str(run_root),
            "--view",
            str(view),
            "--receipt",
            str(receipt),
        ],
        check=True,
    )
    assert not view.exists()
    assert json.loads(receipt.read_text())["file_count_before_cleanup"] == 3


def test_p3_ge_external_capture_uses_xfs_hardlinks_and_marker_cleanup(tmp_path):
    tool = P3 / "manage_ge_external_view.py"
    assets = tmp_path / "assets"
    source = assets / "runtime-weights" / "bundle"
    run_root = tmp_path / "runs" / "run-1"
    evidence = run_root / "evidence"
    source.mkdir(parents=True)
    evidence.mkdir(parents=True)
    weight = source / "weight_abc"
    weight.write_bytes(b"weight")
    (source / "meta.json").write_text(
        json.dumps(
            {
                "hash_to_weight_file": {"abc": str(weight)},
                "hash_to_weight_offset": {"abc": 0},
            }
        ),
        encoding="utf-8",
    )
    (source / "dedup-manifest.json").write_text(
        json.dumps(
            {
                "valid": True,
                "logical_file_count": 1,
                "identity_sha256": "identity",
            }
        ),
        encoding="utf-8",
    )
    capture = assets / ".p3-ge-external-capture-test"

    subprocess.run(
        [
            "python3",
            str(tool),
            "capture-create",
            "--source",
            str(source),
            "--assets-root",
            str(assets),
            "--capture",
            str(capture),
        ],
        check=True,
    )
    capture_weight = capture / weight.name
    assert os.path.samestat(weight.stat(), capture_weight.stat())
    capture_meta = json.loads((capture / "meta.json").read_text())
    assert capture_meta["hash_to_weight_file"]["abc"] == str(capture_weight)

    receipt = evidence / "capture-cleanup.json"
    subprocess.run(
        [
            "python3",
            str(tool),
            "capture-cleanup",
            "--assets-root",
            str(assets),
            "--run-root",
            str(run_root),
            "--capture",
            str(capture),
            "--receipt",
            str(receipt),
        ],
        check=True,
    )
    assert not capture.exists()
    cleanup = json.loads(receipt.read_text())
    assert cleanup["file_count_before_cleanup"] == 3
    assert (
        cleanup["exclusive_allocated_bytes_before_cleanup"]
        < cleanup["allocated_bytes_before_cleanup"]
    )


def test_p3_external_weight_promotion_can_select_oldest_generation(tmp_path):
    source = tmp_path.resolve() / "external-weights"
    source.mkdir()
    older = source / "weight_old"
    newer = source / "weight_new"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    (source / "meta.json").write_text(
        json.dumps(
            {
                "hash_to_weight_file": {
                    "old": str(older),
                    "new": str(newer),
                },
                "hash_to_weight_offset": {"old": 0, "new": 0},
            }
        ),
        encoding="utf-8",
    )

    records, generation_index, generation_count = source_records(
        source, generation_gap_seconds=60, generation="oldest"
    )
    assert [record["hash"] for record in records] == ["old"]
    assert generation_index == 0
    assert generation_count == 2

    all_records, generation_index, generation_count = source_records(
        source, generation_gap_seconds=60, generation="all"
    )
    assert [record["hash"] for record in all_records] == ["new", "old"]
    assert generation_index == -1
    assert generation_count == 2


def test_p3_external_weight_capture_is_xfs_scoped_and_runtime_rooted():
    promoter = (P3 / "promote_ge_external_weights.py").read_text(encoding="utf-8")
    runner = (P3 / "run_graph_oracle_on_910b.sh").read_text(encoding="utf-8")

    assert 'source.name.startswith(".p3-ge-external-capture-")' in promoter
    assert 'result.add_argument("--runtime-root", required=True)' in promoter
    assert 'elif source_is_capture:' in promoter
    assert '"capture_hardlinks": capture_hardlinks' in promoter
    assert "validate_promoted_bundle(" in promoter
    assert "CRUISE_P3_EXTERNAL_CAPTURE_DIR" in runner
    assert "capture-create" in runner
    assert "capture-cleanup" in runner
    assert "CRUISE_P3_REUSE_EXTERNAL_CAPTURE" in runner
    assert 'CRUISE_P3_EXTERNAL_WEIGHT_DIR="${runtime_external_weights}"' in runner

def test_p3_air_promotion_only_allows_equal_length_exact_replacement():
    old = b"/dev/shm/old-prefix"
    new = b"/workspace/new-path"
    assert len(old) == len(new)
    payload = b"before:" + old + b":middle:" + old + b":after"

    result = replace_prefix(payload, old, new, expected_count=2)
    assert result == b"before:" + new + b":middle:" + new + b":after"
    assert len(result) == len(payload)

    try:
        replace_prefix(payload, old, new + b"x", expected_count=2)
    except ValueError as error:
        assert "preserve byte length" in str(error)
    else:
        raise AssertionError("unequal AIR prefix lengths must be rejected")


def test_p3_fia_air_promotion_hardlinks_xfs_staging(tmp_path):
    assets = tmp_path / "assets"
    source = assets / "p3-air384-fia-PLACEHOLDER00000"
    source.mkdir(parents=True)
    first = source / "weight_a"
    second = source / "weight_b"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    prefix = str(source).encode("ascii")
    source_air = source / "qwen_b4_p3_decoder_step.air"
    source_air.write_bytes(b"air:" + prefix + b":" + prefix)
    source_graph = source / "dynamo.pbtxt"
    source_graph.write_text(
        "\n".join(
            [
                "node {",
                '  op: "FileConstant"',
                '  attr { key: "file_path" value {',
                f"    s: 's: \"{first}\"\\n'",
                "  } }",
                "}",
                "node {",
                '  op: "FileConstant"',
                '  attr { key: "file_path" value {',
                f"    s: 's: \"{second}\"\\n'",
                "  } }",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    assert file_constant_paths(source_graph) == [first, second]
    air_digest = hashlib.sha256(source_air.read_bytes()).hexdigest()
    graph_digest = hashlib.sha256(source_graph.read_bytes()).hexdigest()
    (source / "export-result.json").write_text(
        json.dumps(
            {
                "pass": True,
                "air_sha256": air_digest,
                "graph_sha256": graph_digest,
                "external_file_count": 2,
                "eager_page_boundary_exact": True,
                "eager_stock_vllm_semantics_exact": True,
            }
        ),
        encoding="utf-8",
    )
    output = assets / f"p3-air384-fia-{air_digest[:16]}"

    subprocess.run(
        [
            "python3",
            str(P3 / "promote_p3_air_equal_prefix.py"),
            "--source-dir",
            str(source),
            "--source-air",
            str(source_air),
            "--source-graph",
            str(source_graph),
            "--source-result",
            str(source / "export-result.json"),
            "--assets-root",
            str(assets),
            "--output",
            str(output),
            "--expected-external-count",
            "2",
            "--apply",
        ],
        check=True,
    )
    manifest = validate_published(output, source)
    assert manifest["external_file_count"] == 2
    assert os.path.samestat(first.stat(), (output / first.name).stat())
    assert prefix not in (output / "qwen_b4_p3_decoder_step.runtime.air").read_bytes()
    assert prefix not in (output / "dynamo.pbtxt").read_bytes()
    os.chmod(output, 0o755)


def test_p3_owner_uses_persistent_buffer_factory_for_large_kv():
    controller_source = (
        P3 / "controller" / "persistent_decoder_p3.cpp"
    ).read_text(encoding="utf-8")
    host_source = (P3 / "persistent_decoder_p3_host.cpp").read_text(encoding="utf-8")

    assert "FlowBufferFactory::AllocTensor" in controller_source
    assert "context->ToFlowMsg" in controller_source
    assert "EnsureCache(context) != FLOW_FUNC_SUCCESS" in controller_source
    assert "!EnsureCache(context)" not in controller_source
    assert "if (state.rejected > 0) return true;" in host_source
    assert "if (!ingesting) return row.credit > 0;" in controller_source
    assert "left.reserved0 != right.reserved0" in controller_source
    assert "if (!row.active || row.staged) return false;" in controller_source


def test_p3_owner_separates_first_feed_startup_timeout_from_steady_state():
    host_source = (P3 / "persistent_decoder_p3_host.cpp").read_text(
        encoding="utf-8"
    )

    assert "constexpr int32_t kFirstFeedTimeoutMs = 180000;" in host_source
    assert "constexpr int32_t kFeedTimeoutMs = 30000;" in host_source
    assert "bool first_feed_attempted = false;" in host_source
    assert "first_feed_attempted = true;" in host_source
    assert "first_feed ? kFirstFeedTimeoutMs : kFeedTimeoutMs" in host_source
    assert "FeedEvent(session, event, ++transport_seq, timeout_ms)" in host_source
    assert "while" not in host_source[
        host_source.index("auto send =") : host_source.index(
            "std::thread drain", host_source.index("auto send =")
        )
    ]


def test_p3_mini_fia_probe_is_weight_free_and_compares_graph_with_graphpp():
    probe = P3 / "fia_graphpp_probe"
    exporter = (probe / "export_mini_fia.py").read_text(encoding="utf-8")
    host = (probe / "mini_fia_graphpp_probe.cpp").read_text(encoding="utf-8")
    om_host = (probe / "mini_fia_om_probe.cpp").read_text(encoding="utf-8")
    runner = (probe / "run_on_910b.sh").read_text(encoding="utf-8")
    builder = (probe / "build_isolated_opp.sh").read_text(encoding="utf-8")
    auditor = (probe / "audit_isolated_opp.py").read_text(encoding="utf-8")
    hierarchy_patch = (
        probe / "preserve_op_kernel_hierarchy.patch"
    ).read_text(encoding="utf-8")

    assert "npu_fused_infer_attention_score" in exporter
    assert "torchair.ge.custom_op" in exporter
    assert '"dequant_scale_query": None' in exporter
    assert '"learnable_sink": None' in exporter
    assert '"q_start_idx": None' in exporter
    assert '"kv_start_idx": None' in exporter
    assert '"fia_slot_contract_pass"' in exporter
    assert '"fia_input_count"' in exporter
    assert "model-dir" not in exporter
    assert "BATCH = 4" in exporter
    assert "KV_TOKENS = 384" in exporter
    assert 'choices=("dense", "pa-nz")' in exporter
    assert "actual_seq_lengths_kv=[KV_TOKENS] * BATCH" in exporter
    assert "block_table=auxiliary if paged else None" in exporter
    assert "V2-PA-NZ-FIA-EXPORT" in exporter
    assert 'kv_layout == "pa-nz"' in host
    assert "MakeBlockTable" in host
    assert "V2-PA-NZ-FIA-GRAPHPP" in host
    assert "attention.transpose(1, 2).reshape" in exporter
    assert '"mini_fia_graph_pp"' in host
    assert 'mode == "graph"' in host
    assert 'mode == "dataflow"' in host
    assert 'mode != "serialized-dataflow"' in host
    assert "RunGraph" in host
    assert "FeedDataFlowGraph" in host
    assert "LoadFromSerializedModelArray" in host
    assert "mini_fia_serialized_graph_pp" in host
    assert '"ge.externalWeight", "1"' in host
    assert '"CRUISE_MINI_FIA_GE_JIT_COMPILE"' in host
    assert 'config["ge.jit_compile"]' in host
    assert '\\"ge_jit_compile\\"' in host
    assert "EMPTY_EXTERNAL_WEIGHT_DIR" in host
    assert "aclgrphBuildInitialize" in om_host
    assert "aclgrphBuildModel" in om_host
    assert "aclgrphSaveModel" in om_host
    assert 'om_prefix + ".om"' in om_host
    assert '"ge.socVersion", "Ascend910B2"' in om_host
    assert '"ge.exec.precision_mode", "must_keep_origin_dtype"' in om_host
    assert "aclmdlLoadFromMem" in om_host
    assert "aclmdlExecute" in om_host
    assert "ACL_MEMCPY_DEVICE_TO_HOST" in om_host
    assert "output_exact" in om_host
    assert "aclmdlUnload" in om_host
    assert "LoadFromSerializedModelArray" not in om_host
    assert "external-weights" in runner
    assert "CRUISE_MINI_FIA_MODES" in runner
    assert "CRUISE_MINI_FIA_KV_LAYOUT" in runner
    assert "PA-NZ mini FIA supports only graph and dataflow modes" in runner
    assert "CRUISE_MINI_FIA_MODES:-graph dataflow" in runner
    assert "build_isolated_opp.sh" in runner
    assert "source-worktree-status.txt" in runner
    assert "--jit --soc=ascend910b" in builder
    assert "--ops=fused_infer_attention_score" in builder
    assert "--vendor_name=cruise_fia_graph_v2" in builder
    assert "--install-path" in builder
    assert "cpack-staging" in builder
    assert "fia-opp-staging-files.sha256" in builder
    assert "bash ./install.sh --quiet" in builder
    assert "multiple isolated FIA run packages found" in builder
    assert "opp_proxy" in runner
    assert "for component in built-in include lib64 bin Ascend" in runner
    assert "export ASCEND_OPP_PATH=${system_opp}" in runner
    assert "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}" in runner
    assert 'find "${fia_install_root}" -type d -exec chmod u+w {} +' in runner
    assert 'cd "${scratch}"' in runner
    assert 'cd "${source_dir}"' in runner
    assert "tracked ops-transformer source is dirty" in builder
    assert "public_source_commit" in auditor
    assert "isolated_install_root" in auditor
    assert "dynamic_compile_entry_present" in auditor
    assert "cross_family_targets" in auditor
    assert "${_op_name}/op_kernel" in hierarchy_patch
    assert "${_op_depened_name}/op_kernel" in hierarchy_patch
    assert '"${selected_modes[@]}"' in runner
    assert "export_status" in runner
    assert "export-result.json" in runner
    assert '"${evidence}/mini_fia.air"' not in runner
    assert '"${evidence}/mini_fia.om"' not in runner
    assert '"${evidence}/mini_fia.om.sha256"' in runner
    assert "serialized-dataflow requires a preceding om mode" in runner
    assert "CRUISE_CANN_PYTHON_ENV" in runner
    assert "import numpy, te, tbe" in runner
    assert "mode-status.tsv" in runner
    assert "STORAGE_GUARD_MAX_SCRATCH_GIB=1" in runner
    assert "storage_guard_cleanup_scratch" in runner


def test_p3_custom_graphpp_probe_is_isolated_exact_and_public():
    probe = P3 / "custom_graphpp_probe"
    host = (probe / "bf16_graphpp_probe.cpp").read_text(encoding="utf-8")
    runner = (probe / "run_on_910b.sh").read_text(encoding="utf-8")
    verifier = (probe / "verify_probe.py").read_text(encoding="utf-8")
    protocol = (probe / "protocol.md").read_text(encoding="utf-8")

    assert 'GraphPp("bf16_materialize_graph_pp"' in host
    assert 'FlowNode("bf16_materialize_node", 1, 1)' in host
    assert 'mode != "graph" && mode != "dataflow"' in host
    assert "OutputExact(input, outputs)" in host
    assert "aclmdlExecute" not in host
    assert "ModelPp" not in host
    assert "new_op_project_template/custom_op" in runner
    assert "bf16-materialize-attempt56r1" in runner
    assert "export ASCEND_OPP_PATH=${opp_proxy}" in runner
    assert "--install-path=\"${install}\"" in runner
    assert "for mode in graph dataflow" in runner
    assert 'find "${install}" -type d -exec chmod u+w {} +' in runner
    assert "export_status" in runner
    assert "export-result.json" in runner
    assert "cd \"${scratch}\"" in runner
    assert "ASCEND_SLOG_PRINT_TO_STDOUT=0" in runner
    assert "mode_driver_logs=${driver_logs}/${mode}" in runner
    assert '"${evidence}/input.bin"' not in runner
    assert "te_bf16materialize_" in verifier
    assert "output_matches_input" in verifier
    assert "does not implement" in protocol


def test_p3_owner_exercises_lifecycle_and_async_drain_contracts():
    host_source = (P3 / "persistent_decoder_p3_host.cpp").read_text(
        encoding="utf-8"
    )
    runner = (P3 / "run_on_910b2.sh").read_text(encoding="utf-8")

    assert 'scenario == "lifecycle"' in host_source
    assert 'scenario == "b1-eos"' in host_source
    assert 'scenario == "b1-backpressure"' in host_source
    assert "int64_t flags = 0, int64_t credit = -1" in host_source
    assert 'scenario == "b1-full" || scenario == "b1-backpressure" ? 1 : 0' in host_source
    assert "shared.duplicate_acks == 2" in host_source
    assert "stale_it->second == 2" in host_source
    assert "generation_it->second == 1" in host_source
    assert '\\"request_states\\"' in host_source
    assert "std::thread(FetchOutputs" in host_source
    assert "milliseconds(1500)" in host_source
    assert "wait_for_release" in runner
    assert "find \"${driver_logs}\" -type f -size +16M -delete" in runner
    assert 'cp --reflink=auto "${summary}" "${evidence}/summary.json"' in runner
    assert 'mkdir -p "${evidence}/failure-config"' in runner


def test_p3_gate_uses_an_independent_graph_oracle():
    source = (P3 / "p3_graph_oracle.cpp").read_text(encoding="utf-8")

    assert "session->RunGraph(0, inputs, outputs)" in source
    assert "FunctionPp" not in source
    assert "PersistentDecoderP3" not in source
    assert "RowChecksum(key_cache, value_cache, index)" in source
    assert "{28, 12, 128, 4, 128}" in source
    assert '"ge.externalWeight", "1"' in source
    assert '"ge.modelFileNamePrefix", "p3_graph_oracle"' in source
    assert "row.tokens[0] == kDefaultEos" in source
    assert "rows[index].eos_token = kDefaultEos" in source
    assert 'external_weights == run_root + "/.ge-external-view"' in source
    assert 'scenario == "b1-stock"' in source
    assert "{355, 11, 220, 17, 15, 16, 16, 11}" in source


def test_p3_graph_oracle_uses_audited_air_data_order():
    source = (P3 / "p3_graph_oracle.cpp").read_text(encoding="utf-8")

    data_order = (
        "MakeTensor(length_bytes, {4, 1}, ge::DT_INT32),\n"
        "      MakeTensor(key_cache, {28, 12, 128, 4, 128}, ge::DT_BF16),\n"
        "      MakeTensor(slot_bytes, {4}, ge::DT_INT32),\n"
        "      MakeTensor(active_bytes, {4}, ge::DT_INT32),\n"
        "      MakeTensor(block_bytes, {4, 3}, ge::DT_INT32),\n"
        "      MakeTensor(value_cache, {28, 12, 128, 4, 128}, ge::DT_BF16)"
    )
    assert data_order in source
    assert "audited Data-node indexes" in source


def _gate_summary(kind: str) -> dict:
    result = {
        "pass": True,
        "scenario": "b1-short",
        "request_states": [
            {
                "request": 10000,
                "generation": 1,
                "row": 0,
                "commits": 2,
                "retired": True,
                "cancelled": False,
                "final_position": 5,
                "final_page": 0,
                "final_checksum": 123,
                "finish_reason": 2,
                "tokens": [7, 8],
            }
        ],
    }
    if kind == "owner":
        result.update(feed_calls=2, aicore_calls=5)
    else:
        result.update(model_calls=5)
    return result


def test_p3_gate_compares_tokens_terminal_state_and_kv_checksum():
    owner = _gate_summary("owner")
    oracle = _gate_summary("oracle")

    assert compare(owner, oracle)["pass"] is True
    oracle["request_states"][0]["final_checksum"] = 124
    result = compare(owner, oracle)
    assert result["pass"] is False
    assert result["mismatches"] == [
        {
            "request": [10000, 1],
            "field": "final_checksum",
            "owner": 123,
            "oracle": 124,
        }
    ]


def test_p3_allocator_probe_covers_all_candidate_message_paths():
    probe = P3 / "allocator_probe"
    controller = (probe / "controller" / "p3_allocator_probe.cpp").read_text(
        encoding="utf-8"
    )
    host = (probe / "p3_allocator_probe_host.cpp").read_text(encoding="utf-8")

    assert "context->AllocTensorMsg" in controller
    assert "FlowBufferFactory::AllocTensor" in controller
    assert "context->ToFlowMsg" in controller
    assert "context->AllocRawDataMsg" in controller
    assert "context->AllocTensorListMsg" in controller
    assert "{28, 6, 128, 4, 128}" in controller
    assert "{28, 8, 128, 4, 128}" in controller
    assert "{28, 12, 128, 4, 128}" in controller
    assert "tensor-list-2x21" in host
    assert "tensor-msg-2x42" in host
    assert "factory-wrap-2x42" in host
    assert "tensor-msg-4x21" in host
    assert "tensor-list-4x21" in host


def test_p3_allocator_probe_is_function_only_and_runs_cases_independently():
    probe = P3 / "allocator_probe"
    config = (probe / "prepare_probe_config.py").read_text(encoding="utf-8")
    runner = (probe / "run_on_910b2.sh").read_text(encoding="utf-8")

    assert "GraphPp" not in config
    assert '"heavy_load": heavy_load' in config
    assert 'parser.add_argument("--heavy-load", action="store_true")' in config
    assert '"${host_binary}" "${function_config}" "${deploy_config}"' in runner
    assert "for probe_case in" in runner
    assert "wait_for_release" in runner
    assert "CRUISE_P3_ALLOCATOR_CASES" in runner
    assert "CRUISE_P3_ALLOCATOR_HEAVY_LOAD" in runner
    assert "case_tmp=${scratch}/t${case_index}" in runner
