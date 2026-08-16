import json
import sys
from pathlib import Path

import pytest

from experiments.persistent_owner_graph_v2.kv_alias_probe.inspect_kv_alias import (
    inspect_graph_text,
)
from experiments.persistent_owner_graph_v2.kv_alias_probe.prepare_graphpp_config import (
    CACHE_ELEMENTS,
    KEY_BITS,
    VALUE_BITS,
    expected_cache,
)
from experiments.persistent_owner_graph_v2.device_kv_update_probe.inspect_probe_graph import (
    inspect_graph_text as inspect_device_kv_update_graph,
)
from experiments.persistent_owner_graph_v2.device_kv_update_probe.prepare_probe_config import (
    graph_config as device_kv_update_graph_config,
)
from experiments.persistent_owner_graph_v2.paged_kv_order_probe.inspect_probe_graph import (
    inspect_graph_text as inspect_paged_kv_order_graph,
)
from experiments.persistent_owner_graph_v2.paged_kv_order_probe.prepare_probe_config import (
    graph_config as paged_kv_order_graph_config,
)
from experiments.persistent_owner_graph_v2.attention_kv_probe.inspect_probe_graph import (
    inspect_graph_text as inspect_attention_kv_graph,
)
from experiments.persistent_owner_graph_v2.attention_kv_probe.prepare_probe_config import (
    graph_config as attention_kv_graph_config,
    load_abi as load_attention_kv_abi,
    render_header as render_attention_kv_header,
)
from experiments.persistent_owner_graph_v2.attention_kv_probe.verify_probe import (
    main as verify_attention_kv,
    transfer_audit as audit_attention_kv_transfers,
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
            "phase_handoff": "device_handle_continuity",
            "host_copy_bytes": 0,
            "full_cache_materialization_bytes": 0,
            "full_cache_input": True,
            "full_cache_output": False,
            "state_handle_proof": "passed",
            "same_flowmsg_across_calls": True,
            "same_buffer_address_across_calls": True,
            "raw_device_address_abi_used": False,
            "external_refdata_used": False,
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
    state_adr = (
        ROOT
        / "docs"
        / "adr"
        / "0022-retain-shared-state-behind-device-handles.md"
    ).read_text(encoding="utf-8")

    assert "**Controller-Aware Graph Family**" in context
    assert "**Shared Device State Contract**" in context
    assert "ADR 0020 and the P5\nStopped / Unqualified state remain effective" in adr
    assert "private `ModelPp`" in adr
    assert "per-token Host `aclmdlExecute`" in adr
    assert "**Device State Handle**" in context
    assert "FunctionPp-owned `FlowMsg` handles" in state_adr
    assert "raw Device addresses" in state_adr


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
        graph.required_ops["DevicePagedKvUpdate"] == 28
        and "functionpp_owned_state_handle" in graph.required_features
        and "flowmsg_identity_continuity" in graph.required_features
        and "no_external_refdata" in graph.required_features
        and "kv_update_attention_dependency" in graph.required_features
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
    with pytest.raises(GraphFamilyContractError, match="Device-handle input"):
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
        (
            "kv_state",
            "state_handle_proof",
            "pending",
            "kv_state.state_handle_proof",
        ),
        (
            "kv_state",
            "same_flowmsg_across_calls",
            False,
            "kv_state.same_flowmsg_across_calls",
        ),
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


def test_v2_candidate_gate_requires_handle_input_and_forbids_cache_output():
    manifest = load_graph_family(MANIFEST)
    reports = [_inspection(graph) for graph in manifest.graphs]
    reports[1]["kv_state"]["full_cache_input"] = False

    missing_handle = verify_graph_family(manifest, reports)

    assert not missing_handle.passed
    assert any("proven Device handle" in item for item in missing_handle.violations)

    reports = [_inspection(graph) for graph in manifest.graphs]
    reports[1]["kv_state"]["full_cache_output"] = True

    cache_output = verify_graph_family(manifest, reports)

    assert not cache_output.passed
    assert any("full_cache_output" in item for item in cache_output.violations)


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
    assert 'cp "${export_dir}/kv_alias_probe.air"' not in runner
    assert "historical 5% idle-HBM line" in protocol


def test_v2_hardware_policy_uses_stable_baseline_and_relative_recovery():
    policy = (V2 / "hardware_policy.sh").read_text(encoding="utf-8")
    runner = (V2 / "kv_alias_probe" / "run_export_on_910b.sh").read_text(
        encoding="utf-8"
    )

    assert "CRUISE_V2_HBM_STABLE_SAMPLES:-3" in policy
    assert "CRUISE_V2_HBM_STABILITY_TOLERANCE_MB:-64" in policy
    assert "CRUISE_V2_HBM_RECOVERY_TOLERANCE_MB:-64" in policy
    assert "hbm <= baseline_hbm_mb + tolerance_mb" in policy
    assert "No process in device." in policy
    assert "v2_capture_hbm_baseline" in runner
    assert "v2_wait_for_hbm_recovery" in runner


def test_v2_kv_alias_pair_uses_public_graph_and_graphpp_with_exact_oracle():
    probe = V2 / "kv_alias_probe"
    host = (probe / "kv_alias_graphpp_probe.cpp").read_text(encoding="utf-8")
    runner = (probe / "run_graphpp_pair_on_910b.sh").read_text(encoding="utf-8")
    config = (probe / "prepare_graphpp_config.py").read_text(encoding="utf-8")
    verifier = (probe / "verify_graphpp_pair.py").read_text(encoding="utf-8")
    protocol = (probe / "protocol.md").read_text(encoding="utf-8")

    assert 'GraphPp("kv_alias_graph_pp"' in host
    assert 'FlowNode("kv_alias_node", 5, 2)' in host
    assert "session->RunGraph" in host
    assert "session->FeedDataFlowGraph" in host
    assert "session->FetchDataFlowGraph" in host
    assert "SetOutputs({{node, {0, 1}}})" in host
    assert "aclmdlExecute" not in host
    assert "ModelPp" not in host
    assert "for mode in graph dataflow" in runner
    assert "export_status" in runner
    assert "export_status} -eq 0 || ${export_status} -eq 139" in runner
    assert 'cd "${source_dir}"' in runner
    assert "CRUISE_CANN_PYTHON_ENV" in runner
    assert "import numpy, te, tbe" in runner
    assert '"${evidence}/kv_alias_probe.air"' not in runner
    assert "v2_capture_hbm_baseline" in runner
    assert "v2_wait_for_hbm_recovery" in runner
    assert 'struct.pack_into("<H"' in config
    assert "ordinary_graph_matches_graphpp" in verifier
    assert "output_matches_oracle" in verifier
    assert "FIA GraphPp compatibility" in protocol


def test_v2_kv_alias_pair_oracle_has_exact_paged_nz_slots():
    key = expected_cache(KEY_BITS)
    value = expected_cache(VALUE_BITS)

    assert len(key) == CACHE_ELEMENTS * 2
    assert len(value) == CACHE_ELEMENTS * 2
    assert key[:32] == b"\x80\x3f" * 16
    assert value[:32] == b"\x80\xbf" * 16
    assert key[32 : 2048 * 2] == b"\0" * (2048 * 2 - 32)
    second_row_block = 3 * 32 * 128 * 16 * 2
    assert key[second_row_block : second_row_block + 2] == b"\x00\x40"
    assert value[second_row_block : second_row_block + 2] == b"\x00\xc0"


def _device_kv_update_graph(cache_op: str = "Data") -> str:
    return f'''node {{
  name: "cache"
  op: "{cache_op}"
}}
node {{
  name: "metadata"
  op: "Data"
}}
node {{
  name: "update"
  op: "DeviceKvSlotUpdate"
  input: "cache:0"
  input: "metadata:0"
}}
node {{
  name: "output"
  op: "NetOutput"
  input: "update:0"
}}
'''


def test_v2_device_kv_update_structure_rejects_external_refdata():
    accepted = inspect_device_kv_update_graph(_device_kv_update_graph())
    rejected = inspect_device_kv_update_graph(_device_kv_update_graph("RefData"))

    assert accepted["pass"]
    assert accepted["update_input_ops"] == ["Data", "Data"]
    assert accepted["external_refdata_count"] == 0
    assert not accepted["full_cache_graph_output"]
    assert not rejected["pass"]
    assert rejected["external_refdata_count"] == 1


def test_v2_device_kv_update_uses_one_public_flowmsg_across_two_calls():
    probe = V2 / "device_kv_update_probe"
    controller = (probe / "controller" / "device_kv_update_controller.cpp").read_text(
        encoding="utf-8"
    )
    host = (probe / "device_kv_update_probe_host.cpp").read_text(encoding="utf-8")
    protocol = (probe / "protocol.md").read_text(encoding="utf-8")

    assert "FlowBufferFactory::AllocTensor" in controller
    assert '"kv_update_graph_0", {state_cache_, inputs[0]}' in controller
    assert "allocation_count_" in controller
    assert "cache == cache_address_" in controller
    assert "state_cache_.get() == cache_message_address_" in controller
    assert 'AddInvokedClosure("kv_update_graph_0", update)' in host
    assert 'GraphPp("device_kv_update_graph_pp"' in host
    assert 'FlowNode("device_kv_update_controller_node", 1, 1)' in host
    assert "MakeMetadata(1, 0, kFirstValue)" in host
    assert "MakeMetadata(2, kFirstValue, kSecondValue)" in host
    assert "reinterpret_cast<uint64_t>" not in controller
    assert "raw pointer in an integer tensor" in protocol


def test_v2_device_kv_update_has_small_public_io_and_no_cache_output():
    config = device_kv_update_graph_config()
    probe = V2 / "device_kv_update_probe"
    kernel = (probe / "op_kernel" / "device_kv_slot_update.cpp").read_text(
        encoding="utf-8"
    )
    verifier = (probe / "verify_probe.py").read_text(encoding="utf-8")
    runner = (probe / "run_on_910b.sh").read_text(encoding="utf-8")

    assert config["inputs_tensor_desc"] == [
        {"data_type": "DT_BFLOAT16", "shape": [4096]},
        {"data_type": "DT_INT32", "shape": [4]},
    ]
    assert "GM_ADDR gm_cache" in kernel
    assert "cache.SetValue" in kernel
    assert "host_cache_input_bytes" in verifier
    assert "host_cache_output_bytes" in verifier
    assert "exact_kernel_reports" in verifier
    assert "dataflow_launches >= 1" in verifier
    assert "exact sequenced AICore reports" in verifier
    assert "v2_capture_hbm_baseline" in runner
    assert "v2_wait_for_hbm_recovery" in runner
    assert "STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65" in runner
    assert "source-worktree-status.txt" in runner


def _paged_kv_order_graph(ticket_source: str = "update") -> str:
    extra_ticket = ""
    if ticket_source != "update":
        extra_ticket = '''node {
  name: "other_ticket"
  op: "Data"
}
'''
    return f'''node {{
  name: "state"
  op: "Data"
}}
node {{
  name: "metadata"
  op: "Data"
}}
node {{
  name: "update"
  op: "DevicePagedKvUpdate"
  input: "state:0"
  input: "metadata:0"
}}
{extra_ticket}node {{
  name: "reader"
  op: "DevicePagedKvRead"
  input: "state:0"
  input: "metadata:0"
  input: "{ticket_source}:0"
}}
node {{
  name: "output"
  op: "NetOutput"
  input: "reader:0"
}}
'''


def test_v2_paged_kv_order_inspector_requires_explicit_dependency():
    accepted = inspect_paged_kv_order_graph(_paged_kv_order_graph())
    unordered = inspect_paged_kv_order_graph(_paged_kv_order_graph("other_ticket"))
    external = inspect_paged_kv_order_graph(
        _paged_kv_order_graph().replace('op: "Data"', 'op: "RefData"', 1)
    )

    assert accepted["pass"]
    assert accepted["shared_state_input"]
    assert accepted["explicit_update_to_reader_dependency"]
    assert accepted["report_only_graph_output"]
    assert not unordered["pass"]
    assert not unordered["explicit_update_to_reader_dependency"]
    assert not external["pass"]
    assert external["external_refdata_count"] == 1


def test_v2_paged_kv_order_freezes_target_layout_and_small_graph_io():
    config = paged_kv_order_graph_config()
    probe = V2 / "paged_kv_order_probe"
    exporter = (probe / "export_probe.py").read_text(encoding="utf-8")
    protocol = (probe / "protocol.md").read_text(encoding="utf-8")

    assert config["inputs_tensor_desc"] == [
        {"data_type": "DT_BFLOAT16", "shape": [2, 12, 32, 128, 16]},
        {"data_type": "DT_INT32", "shape": [5]},
    ]
    assert "STATE_SHAPE = (2, PHYSICAL_BLOCKS" in exporter
    assert "POSITIONS = (0, 127, 128, 383)" in exporter
    assert "DevicePagedKvUpdate" in exporter
    assert "DevicePagedKvRead" in exporter
    assert "ticket = torch.ops.cruise_device_paged_kv.update" in exporter
    assert "read(state, metadata, ticket)" in exporter
    assert "The graph returns only the reader report" in protocol
    assert (
        "does not establish absence of internal Device-to-Device copies" in protocol
    )


def test_v2_paged_kv_order_has_exact_update_reader_and_full_state_scan():
    probe = V2 / "paged_kv_order_probe"
    update = (probe / "op_kernel" / "device_paged_kv_update.cpp").read_text(
        encoding="utf-8"
    )
    reader = (probe / "op_kernel" / "device_paged_kv_read.cpp").read_text(
        encoding="utf-8"
    )
    controller = (
        probe / "controller" / "paged_kv_order_controller.cpp"
    ).read_text(encoding="utf-8")

    assert "StateIndex" in update
    assert "ExpectedBits(sequence - 1" in update
    assert "state.SetValue(StateIndex" in update
    assert "GM_ADDR gm_ticket" in update
    assert "ticket.GetValue(0) != sequence" in reader
    assert "mismatch_count" in reader
    assert "DataCacheCleanAndInvalid<uint16_t" in reader
    assert "FullStateExact" in controller
    assert "Adler32(state, kStateElements)" in controller
    assert "FlowBufferFactory::AllocTensor" in controller
    assert '"paged_kv_order_graph_0", {state_message_, inputs[0]}' in controller
    assert "state_message_.get() == state_message_address_" in controller
    assert "reinterpret_cast<uint64_t>" not in controller


def test_v2_paged_kv_order_uses_public_graph_graphpp_and_bounded_evidence():
    probe = V2 / "paged_kv_order_probe"
    host = (probe / "paged_kv_order_probe_host.cpp").read_text(encoding="utf-8")
    verifier = (probe / "verify_probe.py").read_text(encoding="utf-8")
    runner = (probe / "run_on_910b.sh").read_text(encoding="utf-8")

    assert 'GraphPp("paged_kv_order_graph_pp"' in host
    assert 'AddInvokedClosure("paged_kv_order_graph_0", order)' in host
    assert 'FlowNode("paged_kv_order_controller_node", 1, 1)' in host
    assert "session->RunGraph" in host
    assert "session->FeedDataFlowGraph" in host
    assert "MakeMetadata(1)" in host
    assert "MakeMetadata(2)" in host
    assert "aclmdlExecute" not in host
    assert "ModelPp" not in host
    assert "explicit_update_to_reader_dependency" in verifier
    assert "exact_full_state_after_each_call" in verifier
    assert "host_cache_input_bytes" in verifier
    assert "host_cache_output_bytes" in verifier
    assert "dataflow_update_launches >= 1" in verifier
    assert "dataflow_read_launches >= 1" in verifier
    assert "v2_capture_hbm_baseline" in runner
    assert "v2_wait_for_hbm_recovery" in runner
    assert "STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65" in runner
    assert '"${evidence}/paged_kv_order_probe.air"' not in runner
    assert "source-worktree-status.txt" in runner


def _attention_kv_data(
    name: str, index: int, data_type: str, shape: tuple[int, ...]
) -> str:
    dimensions = "".join(f"  dim: {dimension}\\n" for dimension in shape)
    descriptor = f"dtype: {data_type}\\nshape {{\\n{dimensions}}}\\n"
    return f'''node {{
  name: "{name}"
  op: "Data"
  attr {{
    key: "index"
    value {{
      s: 'i: {index}\\n'
    }}
  }}
  attr {{
    key: "[i]x"
    value {{
      s: '{descriptor}'
    }}
  }}
  attr {{
    key: "[o]y"
    value {{
      s: '{descriptor}'
    }}
  }}
}}
'''


def _attention_kv_graph(
    ticket_source: str = "update", *, swap_metadata_query: bool = False
) -> str:
    ticket_node = ""
    if ticket_source != "update":
        ticket_node = _attention_kv_data("other_ticket", 6, "DT_INT32", (9,))
    indices = {
        "key_cache": 0,
        "value_cache": 1,
        "metadata": 3 if swap_metadata_query else 2,
        "query": 2 if swap_metadata_query else 3,
        "mask": 4,
        "block_table": 5,
    }
    data_nodes = "".join(
        (
            _attention_kv_data(
                "key_cache", indices["key_cache"], "DT_BF16", (12, 4, 8, 128, 16)
            ),
            _attention_kv_data(
                "value_cache",
                indices["value_cache"],
                "DT_BF16",
                (12, 4, 8, 128, 16),
            ),
            _attention_kv_data(
                "metadata", indices["metadata"], "DT_INT32", (5,)
            ),
            _attention_kv_data(
                "query", indices["query"], "DT_BF16", (4, 28, 1, 128)
            ),
            _attention_kv_data(
                "mask", indices["mask"], "DT_BOOL", (4, 1, 1, 384)
            ),
            _attention_kv_data(
                "block_table", indices["block_table"], "DT_INT32", (4, 3)
            ),
        )
    )
    fia_inputs = [""] * 31
    fia_inputs[0:3] = ["ordered_query", "key_cache", "value_cache"]
    fia_inputs[4] = "mask"
    fia_inputs[14] = "block_table"
    fia_input_text = "".join(
        f'  input: "{name}:0"\n' if name else '  input: ""\n'
        for name in fia_inputs
    )
    return f'''{data_nodes}{ticket_node}node {{
  name: "update"
  op: "DevicePagedKvUpdate"
  input: "key_cache:0"
  input: "value_cache:0"
  input: "metadata:0"
}}
node {{
  name: "ordered_query"
  op: "DeviceQueryAfterKvUpdate"
  input: "query:0"
  input: "{ticket_source}:0"
}}
node {{
  name: "attention"
  op: "FusedInferAttentionScore"
{fia_input_text}}}
node {{
  name: "output"
  op: "NetOutput"
  input: "attention:0"
  input: "update:0"
}}
'''


def test_v2_attention_kv_structure_shares_state_and_orders_fia():
    accepted = inspect_attention_kv_graph(_attention_kv_graph())
    swapped = inspect_attention_kv_graph(
        _attention_kv_graph(swap_metadata_query=True)
    )
    unordered = inspect_attention_kv_graph(_attention_kv_graph("other_ticket"))
    external = inspect_attention_kv_graph(
        _attention_kv_graph().replace(
            'name: "key_cache"\n  op: "Data"',
            'name: "key_cache"\n  op: "RefData"',
        )
    )

    assert accepted["pass"]
    assert accepted["shared_update_and_fia_kv_inputs"]
    assert accepted["explicit_update_to_fia_dependency"]
    assert accepted["compact_attention_and_ticket_outputs"]
    assert not accepted["full_kv_graph_output"]
    assert accepted["external_refdata_count"] == 0
    assert accepted["tensor_move_count"] == 0
    assert accepted["data_input_abi_pass"]
    assert [item["role"] for item in accepted["data_input_abi"]] == [
        "key_cache",
        "value_cache",
        "metadata",
        "query",
        "mask",
        "block_table",
    ]
    assert not swapped["pass"]
    assert not swapped["data_input_abi_pass"]
    assert not unordered["pass"]
    assert not unordered["explicit_update_to_fia_dependency"]
    assert not external["pass"]
    assert external["external_refdata_count"] == 1


def test_v2_attention_kv_config_and_header_follow_validated_air_abi(tmp_path):
    inspected = inspect_attention_kv_graph(_attention_kv_graph())
    abi_path = tmp_path / "graph-abi.json"
    abi_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pass": inspected["data_input_abi_pass"],
                "data_inputs": inspected["data_input_abi"],
            }
        ),
        encoding="utf-8",
    )

    inputs = load_attention_kv_abi(abi_path)
    assert attention_kv_graph_config(inputs)["inputs_tensor_desc"] == [
        {"data_type": "DT_BFLOAT16", "shape": [12, 4, 8, 128, 16]},
        {"data_type": "DT_BFLOAT16", "shape": [12, 4, 8, 128, 16]},
        {"data_type": "DT_INT32", "shape": [5]},
        {"data_type": "DT_BFLOAT16", "shape": [4, 28, 1, 128]},
        {"data_type": "DT_BOOL", "shape": [4, 1, 1, 384]},
        {"data_type": "DT_INT32", "shape": [4, 3]},
    ]
    header = render_attention_kv_header(inputs)
    assert "kInputCount = 6U" in header
    assert "kMetadataInput = 2U" in header
    assert "kQueryInput = 3U" in header

    swapped = json.loads(abi_path.read_text(encoding="utf-8"))
    swapped["data_inputs"][2]["index"] = 3
    swapped["data_inputs"][3]["index"] = 2
    abi_path.write_text(json.dumps(swapped), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or reordered"):
        load_attention_kv_abi(abi_path)

    wrong_shape = json.loads(json.dumps(swapped))
    wrong_shape["data_inputs"][2]["index"] = 2
    wrong_shape["data_inputs"][3]["index"] = 3
    wrong_shape["data_inputs"][3]["shape"] = [4, 28, 2, 128]
    abi_path.write_text(json.dumps(wrong_shape), encoding="utf-8")
    with pytest.raises(ValueError, match="type or shape mismatch"):
        load_attention_kv_abi(abi_path)


def test_v2_attention_kv_transfer_audit_rejects_full_cache_records(tmp_path):
    transfer_log = tmp_path / "transfer.log"
    transfer_log.write_text("rtMemcpyAsync size=28672\n", encoding="utf-8")
    assert audit_attention_kv_transfers(transfer_log) == {
        "record_count": 1,
        "full_cache_record_count": 0,
        "no_full_cache_transfer_record": True,
    }

    transfer_log.write_text(
        "rtMemcpyAsync size=1572864\ncopy task bytes=0x300000\n",
        encoding="utf-8",
    )
    assert audit_attention_kv_transfers(transfer_log) == {
        "record_count": 2,
        "full_cache_record_count": 2,
        "no_full_cache_transfer_record": False,
    }


def test_v2_attention_kv_owner_verifier_separates_registration_from_calls(
    tmp_path, monkeypatch
):
    structure = {
        "pass": True,
        "shared_update_and_fia_kv_inputs": True,
        "explicit_update_to_fia_dependency": True,
        "external_refdata_count": 0,
        "tensor_move_count": 0,
        "compact_attention_and_ticket_outputs": True,
        "full_kv_graph_output": False,
        "data_input_abi_pass": True,
    }
    first = [
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3145728, 0, 0, 14336, 9, 0,
        16256, 16256, 16256, 16256, 0, 4096, 0, 0, 6, 2, 20, 256, 1, 1,
        1, 1,
    ]
    second = [
        2, 0, 1, 2, 1, 1, 1, 1, 1, 1, 3145728, 0, 0, 14336, 9, 16256,
        16384, 16384, 16384, 16384, 0, 4096, 0, 0, 6, 2, 20, 256, 1, 1,
        1, 1,
    ]
    dataflow = {
        "pass": True,
        "exact_two_update_attention_calls": True,
        "abi_input_count": 6,
        "metadata_input_index": 2,
        "query_input_index": 3,
        "allocation_count": 1,
        "graph_call_count": 2,
        "flowmsg_identity_stable": True,
        "buffer_address_stable": True,
        "full_cache_exact_after_each_call": True,
        "attention_observed_update_each_call": True,
        "device_owned_cache_bytes": 3 * 1024 * 1024,
        "host_cache_input_bytes": 0,
        "host_cache_output_bytes": 0,
        "raw_device_address_abi_used": False,
        "external_refdata_used": False,
        "first_summary": first,
        "second_summary": second,
    }
    structure_path = tmp_path / "structure.json"
    dataflow_path = tmp_path / "dataflow.json"
    registration_path = tmp_path / "registrations.log"
    transfer_path = tmp_path / "transfers.log"
    output_path = tmp_path / "verifier.json"
    structure_path.write_text(json.dumps(structure), encoding="utf-8")
    dataflow_path.write_text(json.dumps(dataflow), encoding="utf-8")
    registration_path.write_text(
        "kernel_name=te_devicepagedkvupdate_test\n"
        "kernel_name=te_devicequeryafterkvupdate_test\n"
        "kernel_name=te_fusedinferattentionscore_test\n",
        encoding="utf-8",
    )
    transfer_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_probe.py",
            "--route",
            "owner",
            "--structure",
            str(structure_path),
            "--dataflow-result",
            str(dataflow_path),
            "--dataflow-log",
            str(registration_path),
            "--dataflow-transfer-log",
            str(transfer_path),
            "--output",
            str(output_path),
        ],
    )

    assert verify_attention_kv() == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["pass"]
    assert result["route"] == "owner"
    assert not result["ordinary_graph_evaluated"]
    assert result["ordinary_graph_pass"] is None
    assert result["graphpp_target_tasks_registered"]
    assert result["semantic_graph_calls_exact"]
    assert result["graphpp_registration_counts"] == {
        "update": 1,
        "order": 1,
        "fia": 1,
    }


def test_v2_attention_kv_uses_owned_buffers_narrow_host_io_and_public_routes():
    probe = V2 / "attention_kv_probe"
    controller = (probe / "controller" / "attention_kv_controller.cpp").read_text(
        encoding="utf-8"
    )
    host = (probe / "attention_kv_probe_host.cpp").read_text(encoding="utf-8")
    exporter = (probe / "export_probe.py").read_text(encoding="utf-8")
    runner = (probe / "run_on_910b.sh").read_text(encoding="utf-8")
    verifier = (probe / "verify_probe.py").read_text(encoding="utf-8")
    protocol = (probe / "protocol.md").read_text(encoding="utf-8")

    assert controller.count("FlowBufferFactory::AllocTensor") == 1
    assert "key_message_ = Allocate(context, cache_shape" in controller
    assert "value_message_ = Allocate(context, cache_shape" in controller
    assert "graph_inputs[GraphAbi::kMetadataInput] = inputs[0]" in controller
    assert "graph_inputs[GraphAbi::kQueryInput] = query_message_" in controller
    assert "key_message_.get() == key_message_address_" in controller
    assert "value_message_.get() == value_message_address_" in controller
    assert "reinterpret_cast<uint64_t>" not in controller

    assert 'GraphPp("attention_kv_graph_pp"' in host
    assert 'FlowNode("attention_kv_controller_node", 1, 1)' in host
    assert "session->RunGraph" in host
    assert "tensors_[input_index] = std::move(tensor)" in host
    assert "GraphAbi::kMetadataInput" in host
    assert "GraphAbi::kQueryInput" in host
    assert "desc.SetPlacement(ge::kPlacementDevice)" in host
    assert "tensor.SetData(reinterpret_cast<uint8_t *>(device), bytes" in host
    assert "ge::GEGetErrorMsg()" in host
    assert "aclGetRecentErrMsg()" in host
    assert '\\"host_cache_input_bytes\\": 0' in host
    assert '\\"host_cache_output_bytes\\": 0' in host
    assert "aclmdlExecute" not in host
    assert "ModelPp" not in host

    assert "torch_npu.npu_fused_infer_attention_score" in exporter
    assert '"DevicePagedKvUpdate"' in exporter
    assert '"DeviceQueryAfterKvUpdate"' in exporter
    assert "build_isolated_opp.sh" in runner
    assert "custom_opp_path=${ASCEND_CUSTOM_OPP_PATH}" in runner
    assert (
        "export ASCEND_CUSTOM_OPP_PATH=${custom_opp_path}:${ASCEND_CUSTOM_OPP_PATH}"
        in runner
    )
    assert "CRUISE_V2_KV_ATTENTION_MAX_STAGE:-dataflow" in runner
    assert "CRUISE_ALLOW_DIAGNOSTIC_DIRTY:-0" in runner
    assert '"${allow_diagnostic_dirty}" != 1' in runner
    assert "source-worktree.patch" in runner
    assert "export|graph|owner|dataflow" in runner
    assert 'owner) modes=(dataflow) ;;' in runner
    assert 'dataflow) modes=(graph dataflow) ;;' in runner
    assert '[[ "${max_stage}" == owner ]] && verifier_route=owner' in runner
    assert "graph-transfer-metadata.txt" in runner
    assert "dataflow-transfer-metadata.txt" in runner
    assert 'cp -a "${driver_logs}/${mode}/."' in runner
    assert "driver-log-manifest.tsv" in runner
    assert '"${evidence}/${mode}-error-index.log"' in runner
    assert 'relative=${source_file#"${driver_logs}"/}' in runner
    assert 'head -n 24' not in runner
    assert runner.index('if [[ "${max_stage}" == export ]]') < runner.index(
        'modes=(graph)'
    )
    assert runner.index('if [[ "${mode}" == graph ]]') < runner.index(
        'if [[ "${max_stage}" == graph ]]'
    )
    assert "v2_capture_hbm_baseline" in runner
    assert "v2_wait_for_hbm_recovery" in runner
    assert "STORAGE_GUARD_MAX_IDLE_HBM_PERCENT:-65" in runner

    assert "host_cache_input_bytes" in verifier
    assert "host_cache_output_bytes" in verifier
    assert 'parser.add_argument("--route"' in verifier
    assert "target_tasks_registered" in verifier
    assert "semantic_graph_calls_exact" in verifier
    assert "dataflow_registrations[name] >= 1" in verifier
    assert "dataflow_registrations[name] >= 2" not in verifier
    assert "The owner-only gate passes only when" in protocol
    assert "not one launch per Feed" in protocol
    assert "does not prove" in protocol
