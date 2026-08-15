import json
from pathlib import Path

import pytest

from experiments.persistent_owner_graph_v2.kv_alias_probe.inspect_kv_alias import (
    inspect_graph_text,
)

from vllm_ascend_persistent_owner.graph_family import (
    GraphFamilyContractError,
    GraphPhase,
    OwnerExecutionState,
    load_graph_family,
    verify_graph_family,
)


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "experiments" / "persistent_owner_graph_v2"
MANIFEST = V2 / "graph_family.json"


def _inspection(graph):
    return {
        "schema_version": 1,
        "graph_id": graph.graph_id,
        "extent": {
            "batch_size": graph.extent.batch_size,
            "tokens_per_row": graph.extent.tokens_per_row,
            "attention_capacity_tokens": graph.extent.attention_capacity_tokens,
        },
        "artifact": {
            "sha256": "a" * 64,
            "external_weights_sha256": "b" * 64,
            "bytes": 1024,
        },
        "op_counts": dict(graph.required_ops),
        "features": list(graph.required_features),
        "invocation": {
            "caller": "device_control_plane",
            "process_point": "GraphPp",
            "api": "RunFlowModel",
            "host_token_step_commands": 0,
            "ordinary_graph_load": "passed",
            "graphpp_load": "passed",
        },
        "kv_state": {
            "contract_id": graph.kv_contract_id,
            "access_mode": "paged_in_place",
            "phase_handoff": "device_alias",
            "host_copy_bytes": 0,
            "full_cache_materialization_bytes": 0,
            "full_cache_input": False,
            "full_cache_output": False,
            "alias_proof": "passed",
        },
        "correctness": {
            "ordinary_graph_exact": True,
            "graphpp_exact": True,
            "ordinary_graph_matches_graphpp": True,
        },
        "timing": {"ordinary_graph_ms": 1.0, "graphpp_ms": 1.0},
        "source_identity": {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
            "packages": {
                "cann": "CANN 9.0.0",
                "opp": "ops-transformer 9.0.0",
                "torch": "torch pinned",
                "torch_npu": "torch-npu pinned",
            },
        },
    }


def _write_manifest(tmp_path, mutate):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_v2_language_and_adr_keep_the_historical_stop_boundary():
    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    adr = (
        ROOT
        / "docs"
        / "adr"
        / "0021-preserve-device-control-with-a-graph-family.md"
    ).read_text(encoding="utf-8")

    assert "**Controller-Aware Graph Family**" in context
    assert "**Shared Device State Contract**" in context
    assert "ADR 0020 and the P5\nStopped / Unqualified state remain effective" in adr
    assert "private `ModelPp`" in adr
    assert "per-token Host `aclmdlExecute`" in adr


def test_v2_manifest_freezes_device_selection_and_exact_first_family():
    manifest = load_graph_family(MANIFEST)
    graphs = {graph.phase: graph for graph in manifest.graphs}

    assert manifest.qualification_state == "contract_defined"
    assert manifest.controller.graph_selection_owner == "device_control_plane"
    assert not manifest.controller.host_token_step_control
    assert graphs[GraphPhase.PREFILL].graph_id == "Prefill-B4-L128"
    assert graphs[GraphPhase.PREFILL].extent.tokens_per_row == 128
    assert graphs[GraphPhase.DECODE].graph_id == "Decode-B4"
    assert graphs[GraphPhase.DECODE].extent.attention_capacity_tokens == 384
    assert all(
        graph.required_ops["ScatterPaKvCache"] == 28
        and "kv_refdata_alias" in graph.required_features
        and "no_kv_tensor_move" in graph.required_features
        for graph in manifest.graphs
    )
    assert {graph.kv_contract_id for graph in manifest.graphs} == {
        manifest.shared_state.contract_id
    }


def test_v2_controller_selects_whole_graphs_from_owner_state():
    manifest = load_graph_family(MANIFEST)

    prefill = manifest.select_graph(
        OwnerExecutionState.PREFILL_READY, batch_size=4, tokens_per_row=128
    )
    decode = manifest.select_graph(
        OwnerExecutionState.DECODE_READY, batch_size=4, tokens_per_row=1
    )

    assert prefill.closure_name == "controller_aware_prefill_b4_l128_graph_pp"
    assert decode.closure_name == "controller_aware_decode_b4_graph_pp"
    with pytest.raises(GraphFamilyContractError, match="no unique Device graph"):
        manifest.select_graph(
            OwnerExecutionState.PREFILL_READY, batch_size=4, tokens_per_row=1
        )


def test_v2_manifest_rejects_host_selection_and_unproven_full_cache_io(tmp_path):
    host_path = _write_manifest(
        tmp_path,
        lambda payload: payload["controller"].update(
            {"graph_selection_owner": "host"}
        ),
    )
    with pytest.raises(GraphFamilyContractError, match="Device Control Plane"):
        load_graph_family(host_path)

    cache_path = _write_manifest(
        tmp_path,
        lambda payload: payload["shared_device_state"].update(
            {"full_cache_graph_io_policy": "always_allowed"}
        ),
    )
    with pytest.raises(GraphFamilyContractError, match="proven Device alias"):
        load_graph_family(cache_path)


def test_v2_candidate_gate_accepts_complete_two_graph_evidence():
    manifest = load_graph_family(MANIFEST)
    result = verify_graph_family(manifest, [_inspection(graph) for graph in manifest.graphs])

    assert result.passed
    assert all(result.graph_results.values())
    assert result.violations == ()
    assert "no P5 reopening" in result.as_record()["claim_boundary"]


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("op_counts", "SoftmaxV2", 28, "forbidden SoftmaxV2"),
        ("invocation", "caller", "host", "invocation.caller"),
        ("kv_state", "host_copy_bytes", 1, "kv_state.host_copy_bytes"),
        (
            "kv_state",
            "full_cache_materialization_bytes",
            1,
            "kv_state.full_cache_materialization_bytes",
        ),
        ("kv_state", "alias_proof", "pending", "kv_state.alias_proof"),
    ],
)
def test_v2_candidate_gate_rejects_regressed_compute_or_control_paths(
    section, key, value, message
):
    manifest = load_graph_family(MANIFEST)
    reports = [_inspection(graph) for graph in manifest.graphs]
    reports[1][section][key] = value

    result = verify_graph_family(manifest, reports)

    assert not result.passed
    assert not result.graph_results["Decode-B4"]
    assert any(message in violation for violation in result.violations)


def test_v2_candidate_gate_allows_full_cache_descriptor_only_with_proven_alias():
    manifest = load_graph_family(MANIFEST)
    reports = [_inspection(graph) for graph in manifest.graphs]
    reports[1]["kv_state"]["full_cache_input"] = True
    reports[1]["kv_state"]["full_cache_output"] = True

    result = verify_graph_family(manifest, reports)

    assert result.passed


def test_v2_candidate_gate_rejects_missing_merged_projection():
    manifest = load_graph_family(MANIFEST)
    reports = [_inspection(graph) for graph in manifest.graphs]
    reports[0]["features"].remove("merged_qkv")

    result = verify_graph_family(manifest, reports)

    assert not result.passed
    assert any("merged_qkv" in violation for violation in result.violations)


def test_v2_candidate_gate_rejects_dirty_or_mixed_source_identity():
    manifest = load_graph_family(MANIFEST)
    reports = [_inspection(graph) for graph in manifest.graphs]
    reports[0]["source_identity"]["git_dirty"] = True

    dirty = verify_graph_family(manifest, reports)

    assert not dirty.passed
    assert any("git_dirty" in violation for violation in dirty.violations)

    reports = [_inspection(graph) for graph in manifest.graphs]
    reports[1]["source_identity"]["packages"]["cann"] = "different"

    mixed = verify_graph_family(manifest, reports)

    assert not mixed.passed
    assert all(not passed for passed in mixed.graph_results.values())
    assert "Prefill and Decode source/package identities differ" in mixed.violations


def test_v2_candidate_gate_rejects_inexact_graph_execution():
    manifest = load_graph_family(MANIFEST)
    reports = [_inspection(graph) for graph in manifest.graphs]
    reports[0]["correctness"]["graphpp_exact"] = False

    result = verify_graph_family(manifest, reports)

    assert not result.passed
    assert any("graphpp_exact" in violation for violation in result.violations)


def test_v2_checked_in_state_has_no_synthetic_hardware_evidence():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = (V2 / "protocol.md").read_text(encoding="utf-8")

    assert payload["qualification_state"] == "contract_defined"
    assert all(
        graph["qualification_state"] == "contract_defined"
        for graph in payload["graphs"]
    )
    assert "Contract validation alone does not satisfy" in protocol
    assert "P5 remains Stopped /\nUnqualified" in protocol


def _kv_alias_graph(cache_op: str) -> str:
    return f'''node {{
  name: "key"
  op: "Data"
}}
node {{
  name: "key_cache"
  op: "{cache_op}"
}}
node {{
  name: "slots"
  op: "Data"
}}
node {{
  name: "value"
  op: "Data"
}}
node {{
  name: "value_cache"
  op: "{cache_op}"
}}
node {{
  name: "scatter"
  op: "ScatterPaKvCache"
  input: "key:0"
  input: "key_cache:0"
  input: "slots:0"
  input: "value:0"
  input: "value_cache:0"
}}
'''


def test_v2_kv_alias_inspector_requires_direct_refdata_inputs():
    accepted = inspect_graph_text(_kv_alias_graph("RefData"))
    copied = inspect_graph_text(_kv_alias_graph("TensorMove"))

    assert accepted["pass"]
    assert accepted["kv_refdata_alias"]
    assert accepted["no_kv_tensor_move"]
    assert not copied["pass"]
    assert copied["kv_cache_input_ops"] == ["TensorMove", "TensorMove"]


def test_v2_kv_alias_export_uses_public_paged_update_and_target_extent():
    probe = ROOT / "experiments" / "persistent_owner_graph_v2" / "kv_alias_probe"
    exporter = (probe / "export_kv_alias.py").read_text(encoding="utf-8")
    runner = (probe / "run_export_on_910b.sh").read_text(encoding="utf-8")
    protocol = (probe / "protocol.md").read_text(encoding="utf-8")

    assert "torch_npu.npu_scatter_pa_kv_cache" in exporter
    assert "PHYSICAL_BLOCKS = BATCH_SIZE * BLOCKS_PER_ROW" in exporter
    assert "BLOCK_SIZE = 128" in exporter
    assert "torch.where" not in exporter
    assert "torch.index_select" not in exporter
    assert "STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65" in runner
    assert "STORAGE_GUARD_NPU_STABLE_SAMPLES=3" in runner
    assert "historical 5% idle-HBM line" in protocol
