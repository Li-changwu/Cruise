"""Contracts for a Device-selected family of optimized model graphs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CONTRACT_DEFINED = "contract_defined"
SUPPORTED_PROCESS_POINT = "GraphPp"
SUPPORTED_INVOCATION_API = "RunFlowModel"
DEVICE_CONTROL_PLANE = "device_control_plane"
PAGED_IN_PLACE = "paged_in_place"
DEVICE_HANDLE_CONTINUITY = "device_handle_continuity"
PROOF_PASSED = "passed"
DEVICE_HANDLE_INPUT_ONLY = "device_handle_input_only"


class GraphFamilyContractError(ValueError):
    """Raised when a graph-family contract is incomplete or contradictory."""


class GraphPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class OwnerExecutionState(str, Enum):
    PREFILL_READY = "prefill_ready"
    DECODE_READY = "decode_ready"


@dataclass(frozen=True)
class GraphExtent:
    batch_size: int
    tokens_per_row: int
    attention_capacity_tokens: int


@dataclass(frozen=True)
class GraphSpec:
    graph_id: str
    closure_name: str
    phase: GraphPhase
    extent: GraphExtent
    qualification_state: str
    required_ops: Mapping[str, int]
    forbidden_ops: tuple[str, ...]
    required_features: tuple[str, ...]
    kv_contract_id: str


@dataclass(frozen=True)
class ControllerContract:
    authority: str
    graph_selection_owner: str
    process_point: str
    invocation_api: str
    closure_lifetime: str
    host_token_step_control: bool


@dataclass(frozen=True)
class SharedDeviceStateContract:
    contract_id: str
    owner: str
    batch_size: int
    page_size_tokens: int
    pages_per_row: int
    capacity_tokens_per_row: int
    access_mode: str
    phase_handoff: str
    host_copy_allowed: bool
    full_cache_graph_io_policy: str
    proof_required: str


@dataclass(frozen=True)
class GraphFamilyManifest:
    family_id: str
    model_id: str
    model_revision: str
    qualification_state: str
    controller: ControllerContract
    shared_state: SharedDeviceStateContract
    graphs: tuple[GraphSpec, ...]

    def select_graph(
        self,
        owner_state: OwnerExecutionState,
        *,
        batch_size: int,
        tokens_per_row: int,
    ) -> GraphSpec:
        """Select an exact graph from Device-owned execution state."""

        phase_by_state = {
            OwnerExecutionState.PREFILL_READY: GraphPhase.PREFILL,
            OwnerExecutionState.DECODE_READY: GraphPhase.DECODE,
        }
        try:
            phase = phase_by_state[OwnerExecutionState(owner_state)]
        except (KeyError, ValueError) as exc:
            raise GraphFamilyContractError(
                f"unsupported Owner execution state: {owner_state!r}"
            ) from exc

        matches = [
            graph
            for graph in self.graphs
            if graph.phase is phase
            and graph.extent.batch_size == batch_size
            and graph.extent.tokens_per_row == tokens_per_row
        ]
        if len(matches) != 1:
            raise GraphFamilyContractError(
                "no unique Device graph for "
                f"state={phase.value}, batch_size={batch_size}, "
                f"tokens_per_row={tokens_per_row}"
            )
        return matches[0]


@dataclass(frozen=True)
class GraphFamilyVerification:
    family_id: str
    passed: bool
    graph_results: Mapping[str, bool]
    violations: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "gate": "controller-aware-graph-family-v2-candidate",
            "family_id": self.family_id,
            "pass": self.passed,
            "graph_results": dict(self.graph_results),
            "violations": list(self.violations),
            "claim_boundary": (
                "Manifest, graph structure, public invocation, and shared Device "
                "state evidence only; no P5 reopening or performance claim."
            ),
        }


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GraphFamilyContractError(f"{location} must be an object")
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise GraphFamilyContractError(f"{location} must be an array")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise GraphFamilyContractError(f"{location} must be a non-empty string")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise GraphFamilyContractError(f"{location} must be a boolean")
    return value


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GraphFamilyContractError(f"{location} must be a positive integer")
    return value


def _non_negative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GraphFamilyContractError(f"{location} must be a non-negative integer")
    return value


def _positive_number(value: Any, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise GraphFamilyContractError(f"{location} must be a positive finite number")
    return float(value)


def _sha256_string(value: Any, location: str) -> str:
    digest = _string(value, location)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise GraphFamilyContractError(f"{location} must be a lowercase SHA-256")
    return digest


def _require_keys(
    value: Mapping[str, Any], required: set[str], location: str
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unknown={extra}")
        raise GraphFamilyContractError(f"{location} fields invalid: {', '.join(details)}")


def _string_tuple(value: Any, location: str) -> tuple[str, ...]:
    items = _sequence(value, location)
    result = tuple(_string(item, f"{location}[]") for item in items)
    if len(result) != len(set(result)):
        raise GraphFamilyContractError(f"{location} contains duplicates")
    return result


def _parse_graph(value: Any, index: int) -> GraphSpec:
    location = f"graphs[{index}]"
    root = _mapping(value, location)
    required = {
        "graph_id",
        "closure_name",
        "phase",
        "extent",
        "qualification_state",
        "required_ops",
        "forbidden_ops",
        "required_features",
        "kv_contract_id",
    }
    _require_keys(root, required, location)
    extent_value = _mapping(root["extent"], f"{location}.extent")
    _require_keys(
        extent_value,
        {"batch_size", "tokens_per_row", "attention_capacity_tokens"},
        f"{location}.extent",
    )
    required_ops_value = _mapping(root["required_ops"], f"{location}.required_ops")
    required_ops = {
        _string(name, f"{location}.required_ops key"): _positive_int(
            count, f"{location}.required_ops.{name}"
        )
        for name, count in required_ops_value.items()
    }
    if not required_ops:
        raise GraphFamilyContractError(f"{location}.required_ops cannot be empty")
    try:
        phase = GraphPhase(_string(root["phase"], f"{location}.phase"))
    except ValueError as exc:
        raise GraphFamilyContractError(f"{location}.phase is unsupported") from exc
    return GraphSpec(
        graph_id=_string(root["graph_id"], f"{location}.graph_id"),
        closure_name=_string(root["closure_name"], f"{location}.closure_name"),
        phase=phase,
        extent=GraphExtent(
            batch_size=_positive_int(
                extent_value["batch_size"], f"{location}.extent.batch_size"
            ),
            tokens_per_row=_positive_int(
                extent_value["tokens_per_row"],
                f"{location}.extent.tokens_per_row",
            ),
            attention_capacity_tokens=_positive_int(
                extent_value["attention_capacity_tokens"],
                f"{location}.extent.attention_capacity_tokens",
            ),
        ),
        qualification_state=_string(
            root["qualification_state"], f"{location}.qualification_state"
        ),
        required_ops=required_ops,
        forbidden_ops=_string_tuple(
            root["forbidden_ops"], f"{location}.forbidden_ops"
        ),
        required_features=_string_tuple(
            root["required_features"], f"{location}.required_features"
        ),
        kv_contract_id=_string(root["kv_contract_id"], f"{location}.kv_contract_id"),
    )


def _parse_manifest(payload: Any) -> GraphFamilyManifest:
    root = _mapping(payload, "manifest")
    required = {
        "schema_version",
        "family_id",
        "model",
        "qualification_state",
        "controller",
        "shared_device_state",
        "graphs",
    }
    _require_keys(root, required, "manifest")
    if root["schema_version"] != SCHEMA_VERSION:
        raise GraphFamilyContractError(
            f"unsupported schema_version: {root['schema_version']!r}"
        )

    model = _mapping(root["model"], "model")
    _require_keys(model, {"id", "revision"}, "model")

    controller_value = _mapping(root["controller"], "controller")
    _require_keys(
        controller_value,
        {
            "authority",
            "graph_selection_owner",
            "process_point",
            "invocation_api",
            "closure_lifetime",
            "host_token_step_control",
        },
        "controller",
    )
    controller = ControllerContract(
        authority=_string(controller_value["authority"], "controller.authority"),
        graph_selection_owner=_string(
            controller_value["graph_selection_owner"],
            "controller.graph_selection_owner",
        ),
        process_point=_string(
            controller_value["process_point"], "controller.process_point"
        ),
        invocation_api=_string(
            controller_value["invocation_api"], "controller.invocation_api"
        ),
        closure_lifetime=_string(
            controller_value["closure_lifetime"], "controller.closure_lifetime"
        ),
        host_token_step_control=_boolean(
            controller_value["host_token_step_control"],
            "controller.host_token_step_control",
        ),
    )

    state_value = _mapping(root["shared_device_state"], "shared_device_state")
    _require_keys(
        state_value,
        {
            "contract_id",
            "owner",
            "batch_size",
            "page_size_tokens",
            "pages_per_row",
            "capacity_tokens_per_row",
            "access_mode",
            "phase_handoff",
            "host_copy_allowed",
            "full_cache_graph_io_policy",
            "proof_required",
        },
        "shared_device_state",
    )
    shared_state = SharedDeviceStateContract(
        contract_id=_string(
            state_value["contract_id"], "shared_device_state.contract_id"
        ),
        owner=_string(state_value["owner"], "shared_device_state.owner"),
        batch_size=_positive_int(
            state_value["batch_size"], "shared_device_state.batch_size"
        ),
        page_size_tokens=_positive_int(
            state_value["page_size_tokens"],
            "shared_device_state.page_size_tokens",
        ),
        pages_per_row=_positive_int(
            state_value["pages_per_row"], "shared_device_state.pages_per_row"
        ),
        capacity_tokens_per_row=_positive_int(
            state_value["capacity_tokens_per_row"],
            "shared_device_state.capacity_tokens_per_row",
        ),
        access_mode=_string(
            state_value["access_mode"], "shared_device_state.access_mode"
        ),
        phase_handoff=_string(
            state_value["phase_handoff"], "shared_device_state.phase_handoff"
        ),
        host_copy_allowed=_boolean(
            state_value["host_copy_allowed"],
            "shared_device_state.host_copy_allowed",
        ),
        full_cache_graph_io_policy=_string(
            state_value["full_cache_graph_io_policy"],
            "shared_device_state.full_cache_graph_io_policy",
        ),
        proof_required=_string(
            state_value["proof_required"], "shared_device_state.proof_required"
        ),
    )

    graphs = tuple(
        _parse_graph(item, index)
        for index, item in enumerate(_sequence(root["graphs"], "graphs"))
    )
    manifest = GraphFamilyManifest(
        family_id=_string(root["family_id"], "family_id"),
        model_id=_string(model["id"], "model.id"),
        model_revision=_string(model["revision"], "model.revision"),
        qualification_state=_string(
            root["qualification_state"], "qualification_state"
        ),
        controller=controller,
        shared_state=shared_state,
        graphs=graphs,
    )
    _validate_manifest_contract(manifest)
    return manifest


def _validate_manifest_contract(manifest: GraphFamilyManifest) -> None:
    violations = []
    controller = manifest.controller
    state = manifest.shared_state
    if manifest.qualification_state != CONTRACT_DEFINED:
        violations.append("a checked-in V2 manifest must start at contract_defined")
    if controller.authority != "persistent_device_model_owner":
        violations.append("controller authority must remain on the Persistent Owner")
    if controller.graph_selection_owner != DEVICE_CONTROL_PLANE:
        violations.append("graph selection must remain on the Device Control Plane")
    if controller.process_point != SUPPORTED_PROCESS_POINT:
        violations.append("only public GraphPp process points are allowed")
    if controller.invocation_api != SUPPORTED_INVOCATION_API:
        violations.append("the Device controller must invoke graphs with RunFlowModel")
    if controller.closure_lifetime != "model_load":
        violations.append("graph closures must be created at model load")
    if controller.host_token_step_control:
        violations.append("Host token-step control is forbidden")
    if state.owner != "persistent_device_model_owner":
        violations.append("the Persistent Owner must own shared KV state")
    if state.access_mode != PAGED_IN_PLACE:
        violations.append("shared KV must use paged_in_place access")
    if state.phase_handoff != DEVICE_HANDLE_CONTINUITY:
        violations.append("Prefill/Decode KV handoff must preserve a Device handle")
    if state.host_copy_allowed:
        violations.append("Host KV copies are forbidden")
    if state.full_cache_graph_io_policy != DEVICE_HANDLE_INPUT_ONLY:
        violations.append("shared KV permits only a proven Device-handle input")
    if state.proof_required != "before_graphpp_candidate_gate":
        violations.append(
            "the Device State Handle proof must precede the GraphPp candidate gate"
        )
    if state.capacity_tokens_per_row != state.page_size_tokens * state.pages_per_row:
        violations.append("KV capacity must equal page size times pages per row")

    graph_ids = [graph.graph_id for graph in manifest.graphs]
    closure_names = [graph.closure_name for graph in manifest.graphs]
    if len(graph_ids) != len(set(graph_ids)):
        violations.append("graph_id values must be unique")
    if len(closure_names) != len(set(closure_names)):
        violations.append("closure_name values must be unique")
    by_phase = {phase: [] for phase in GraphPhase}
    for graph in manifest.graphs:
        by_phase[graph.phase].append(graph)
        if graph.qualification_state != CONTRACT_DEFINED:
            violations.append(f"{graph.graph_id} must start at contract_defined")
        if graph.kv_contract_id != state.contract_id:
            violations.append(f"{graph.graph_id} uses a different KV contract")
        if graph.extent.batch_size != state.batch_size:
            violations.append(f"{graph.graph_id} batch does not match shared KV")
        if "FusedInferAttentionScore" not in graph.required_ops:
            violations.append(f"{graph.graph_id} does not require fused attention")
        if graph.required_ops.get("DevicePagedKvUpdate") != 28:
            violations.append(
                f"{graph.graph_id} must require 28 DevicePagedKvUpdate nodes"
            )
        for forbidden in ("SoftmaxV2", "BatchMatMul"):
            if forbidden not in graph.forbidden_ops:
                violations.append(f"{graph.graph_id} must forbid {forbidden}")
        for feature in (
            "merged_qkv",
            "merged_gate_up",
            "native_norm",
            "native_rotary",
            "fused_attention",
            "paged_kv_update",
            "functionpp_owned_state_handle",
            "flowmsg_identity_continuity",
            "no_external_refdata",
            "compact_state_report_only",
            "kv_update_attention_dependency",
            "no_kv_tensor_move",
        ):
            if feature not in graph.required_features:
                violations.append(f"{graph.graph_id} must require {feature}")

    if len(by_phase[GraphPhase.PREFILL]) != 1:
        violations.append("the first family must contain exactly one Prefill graph")
    elif by_phase[GraphPhase.PREFILL][0].extent != GraphExtent(4, 128, 128):
        violations.append("the first Prefill graph must be exactly B4/L128")
    if len(by_phase[GraphPhase.DECODE]) != 1:
        violations.append("the first family must contain exactly one Decode graph")
    elif by_phase[GraphPhase.DECODE][0].extent != GraphExtent(4, 1, 384):
        violations.append("the first Decode graph must be exactly B4/K384")

    if violations:
        raise GraphFamilyContractError("; ".join(violations))


def load_graph_family(path: Path) -> GraphFamilyManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphFamilyContractError(f"cannot load graph-family manifest: {exc}") from exc
    return _parse_manifest(payload)


def _inspection_by_id(inspections: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(inspections):
        report = _mapping(raw, f"inspections[{index}]")
        graph_id = _string(report.get("graph_id"), f"inspections[{index}].graph_id")
        if graph_id in indexed:
            raise GraphFamilyContractError(f"duplicate inspection for {graph_id}")
        indexed[graph_id] = report
    return indexed


def verify_graph_family(
    manifest: GraphFamilyManifest,
    inspections: Sequence[Mapping[str, Any]],
) -> GraphFamilyVerification:
    """Check export, public invocation, and shared-state evidence for each graph."""

    reports = _inspection_by_id(inspections)
    expected_ids = {graph.graph_id for graph in manifest.graphs}
    violations: list[str] = []
    results: dict[str, bool] = {}
    unexpected = sorted(set(reports) - expected_ids)
    if unexpected:
        violations.append(f"unexpected graph inspections: {unexpected}")

    observed_source_identities: list[Mapping[str, Any]] = []
    for graph in manifest.graphs:
        prefix = graph.graph_id
        report = reports.get(graph.graph_id)
        graph_violations: list[str] = []
        if report is None:
            graph_violations.append("inspection is missing")
        else:
            if report.get("schema_version") != SCHEMA_VERSION:
                graph_violations.append("inspection schema_version is unsupported")
            extent = _mapping(report.get("extent"), f"{prefix}.extent")
            observed_extent = (
                extent.get("batch_size"),
                extent.get("tokens_per_row"),
                extent.get("attention_capacity_tokens"),
            )
            expected_extent = (
                graph.extent.batch_size,
                graph.extent.tokens_per_row,
                graph.extent.attention_capacity_tokens,
            )
            if observed_extent != expected_extent:
                graph_violations.append(
                    f"extent {observed_extent} does not match {expected_extent}"
                )

            op_counts = _mapping(report.get("op_counts"), f"{prefix}.op_counts")
            for op_name, minimum in graph.required_ops.items():
                observed = op_counts.get(op_name, 0)
                if (
                    isinstance(observed, bool)
                    or not isinstance(observed, int)
                    or observed < minimum
                ):
                    graph_violations.append(
                        f"{op_name} count {observed!r} is below {minimum}"
                    )
            for op_name in graph.forbidden_ops:
                observed = op_counts.get(op_name, 0)
                if observed != 0:
                    graph_violations.append(
                        f"forbidden {op_name} count is {observed!r}"
                    )

            features = report.get("features")
            if not isinstance(features, list) or not all(
                isinstance(item, str) for item in features
            ):
                graph_violations.append("features must be an array of strings")
                observed_features: set[str] = set()
            else:
                observed_features = set(features)
            missing_features = sorted(set(graph.required_features) - observed_features)
            if missing_features:
                graph_violations.append(
                    f"required structural features missing: {missing_features}"
                )

            invocation = _mapping(report.get("invocation"), f"{prefix}.invocation")
            expected_invocation = {
                "caller": DEVICE_CONTROL_PLANE,
                "process_point": SUPPORTED_PROCESS_POINT,
                "api": SUPPORTED_INVOCATION_API,
                "host_token_step_commands": 0,
                "ordinary_graph_load": PROOF_PASSED,
                "graphpp_load": PROOF_PASSED,
            }
            for key, expected in expected_invocation.items():
                if invocation.get(key) != expected:
                    graph_violations.append(
                        f"invocation.{key} must be {expected!r}"
                    )

            kv_state = _mapping(report.get("kv_state"), f"{prefix}.kv_state")
            expected_state = {
                "contract_id": manifest.shared_state.contract_id,
                "access_mode": PAGED_IN_PLACE,
                "phase_handoff": DEVICE_HANDLE_CONTINUITY,
                "host_copy_bytes": 0,
                "full_cache_materialization_bytes": 0,
                "state_handle_proof": PROOF_PASSED,
                "same_flowmsg_across_calls": True,
                "same_buffer_address_across_calls": True,
                "raw_device_address_abi_used": False,
                "external_refdata_used": False,
            }
            for key, expected in expected_state.items():
                if kv_state.get(key) != expected:
                    graph_violations.append(f"kv_state.{key} must be {expected!r}")
            if kv_state.get("full_cache_input") is not True:
                graph_violations.append(
                    "kv_state.full_cache_input must be a proven Device handle"
                )
            if kv_state.get("full_cache_output") is not False:
                graph_violations.append("kv_state.full_cache_output must be false")

            artifact = _mapping(report.get("artifact"), f"{prefix}.artifact")
            _sha256_string(artifact.get("sha256"), f"{prefix}.artifact.sha256")
            _sha256_string(
                artifact.get("external_weights_sha256"),
                f"{prefix}.artifact.external_weights_sha256",
            )
            _non_negative_int(artifact.get("bytes"), f"{prefix}.artifact.bytes")
            if artifact.get("bytes") == 0:
                graph_violations.append("artifact.bytes must be greater than zero")

            correctness = _mapping(
                report.get("correctness"), f"{prefix}.correctness"
            )
            for key in (
                "ordinary_graph_exact",
                "graphpp_exact",
                "ordinary_graph_matches_graphpp",
            ):
                if correctness.get(key) is not True:
                    graph_violations.append(f"correctness.{key} must be true")

            timing = _mapping(report.get("timing"), f"{prefix}.timing")
            _positive_number(
                timing.get("ordinary_graph_ms"),
                f"{prefix}.timing.ordinary_graph_ms",
            )
            _positive_number(
                timing.get("graphpp_ms"), f"{prefix}.timing.graphpp_ms"
            )

            source_identity = _mapping(
                report.get("source_identity"), f"{prefix}.source_identity"
            )
            commit = _string(
                source_identity.get("git_commit"),
                f"{prefix}.source_identity.git_commit",
            )
            if len(commit) != 40 or any(
                character not in "0123456789abcdef" for character in commit
            ):
                graph_violations.append(
                    "source_identity.git_commit must be a lowercase 40-hex commit"
                )
            if source_identity.get("git_dirty") is not False:
                graph_violations.append("source_identity.git_dirty must be false")
            if source_identity.get("model_revision") != manifest.model_revision:
                graph_violations.append(
                    "source_identity.model_revision does not match the manifest"
                )
            packages = _mapping(
                source_identity.get("packages"),
                f"{prefix}.source_identity.packages",
            )
            for package in ("cann", "opp", "torch", "torch_npu"):
                try:
                    _string(
                        packages.get(package),
                        f"{prefix}.source_identity.packages.{package}",
                    )
                except GraphFamilyContractError as exc:
                    graph_violations.append(str(exc))
            observed_source_identities.append(source_identity)

        results[graph.graph_id] = not graph_violations
        violations.extend(f"{prefix}: {item}" for item in graph_violations)

    if observed_source_identities and any(
        identity != observed_source_identities[0]
        for identity in observed_source_identities[1:]
    ):
        violations.append("Prefill and Decode source/package identities differ")
        results = {graph_id: False for graph_id in results}

    return GraphFamilyVerification(
        family_id=manifest.family_id,
        passed=bool(results) and all(results.values()) and not violations,
        graph_results=results,
        violations=tuple(violations),
    )


def _load_inspection(path: Path) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphFamilyContractError(f"cannot load inspection {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        manifest = load_graph_family(args.manifest)
        if args.inspection:
            verification = verify_graph_family(
                manifest, [_load_inspection(path) for path in args.inspection]
            )
            record = verification.as_record()
            exit_code = 0 if verification.passed else 1
        else:
            record = {
                "schema_version": SCHEMA_VERSION,
                "gate": "controller-aware-graph-family-v2-contract",
                "family_id": manifest.family_id,
                "pass": True,
                "qualification_state": manifest.qualification_state,
                "claim_boundary": (
                    "Contract validation only; graph artifacts, target hardware, "
                    "correctness, Device State Handle continuity, and performance "
                    "are unproven."
                ),
            }
            exit_code = 0
    except GraphFamilyContractError as exc:
        record = {
            "schema_version": SCHEMA_VERSION,
            "gate": "controller-aware-graph-family-v2-contract",
            "pass": False,
            "error": str(exc),
        }
        exit_code = 2

    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
