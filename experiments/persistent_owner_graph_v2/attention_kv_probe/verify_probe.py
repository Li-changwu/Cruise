#!/usr/bin/env python3
"""Verify the combined FunctionPp-owned PA-NZ update-to-FIA safety gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "update": re.compile(r"kernel_name=te_devicepagedkvupdate_", re.IGNORECASE),
    "order": re.compile(r"kernel_name=te_devicequeryafterkvupdate_", re.IGNORECASE),
    "fia": re.compile(r"kernel_name=te_fusedinferattentionscore_", re.IGNORECASE),
}
FULL_CACHE_SIZES = ("1572864", "3145728", "0x180000", "0x300000")


def read_json(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def registration_counts(path: Path | None) -> dict[str, int]:
    text = (
        path.read_text(encoding="utf-8", errors="replace")
        if path is not None and path.is_file()
        else ""
    )
    return {name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()}


def transfer_audit(path: Path) -> dict[str, int | bool]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    lines = [line for line in text.splitlines() if line.strip()]
    full_cache = [
        line
        for line in lines
        if any(size in line.lower() for size in FULL_CACHE_SIZES)
    ]
    return {
        "record_count": len(lines),
        "full_cache_record_count": len(full_cache),
        "no_full_cache_transfer_record": not full_cache,
    }


def exact_summaries(dataflow: dict) -> bool:
    first = dataflow.get("first_summary", [])
    second = dataflow.get("second_summary", [])
    return (
        len(first) == 32
        and len(second) == 32
        and first[0] == 1
        and second[0] == 2
        and first[1:10] == [0, 1, 1, 1, 1, 1, 1, 1, 1]
        and second[1:10] == [0, 1, 2, 1, 1, 1, 1, 1, 1]
        and first[11:13] == [0, 0]
        and second[11:13] == [0, 0]
        and first[18:22] == [16256, 16256, 0, 4096]
        and second[18:22] == [16384, 16384, 0, 4096]
        and first[22:24] == [0, 0]
        and second[22:24] == [0, 0]
        and first[28:32] == [1, 1, 1, 1]
        and second[28:32] == [1, 1, 1, 1]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("combined", "owner"), default="combined")
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--graph-result", type=Path)
    parser.add_argument("--graph-log", type=Path)
    parser.add_argument("--graph-transfer-log", type=Path)
    parser.add_argument("--dataflow-result", type=Path, required=True)
    parser.add_argument("--dataflow-log", type=Path, required=True)
    parser.add_argument("--dataflow-transfer-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.route == "combined" and any(
        path is None
        for path in (args.graph_result, args.graph_log, args.graph_transfer_log)
    ):
        parser.error("combined route requires all ordinary Graph evidence")

    structure = read_json(args.structure)
    graph = read_json(args.graph_result)
    dataflow = read_json(args.dataflow_result)
    graph_registrations = registration_counts(args.graph_log)
    dataflow_registrations = registration_counts(args.dataflow_log)
    graph_transfers = (
        transfer_audit(args.graph_transfer_log)
        if args.graph_transfer_log is not None
        else None
    )
    dataflow_transfers = transfer_audit(args.dataflow_transfer_log)
    summaries_exact = exact_summaries(dataflow)
    target_tasks_registered = all(
        dataflow_registrations[name] >= 1 for name in PATTERNS
    )
    semantic_graph_calls_exact = (
        dataflow.get("exact_two_update_attention_calls") is True
        and dataflow.get("graph_call_count") == 2
        and summaries_exact
    )

    structure_pass = (
        structure.get("pass") is True
        and structure.get("shared_update_and_fia_kv_inputs") is True
        and structure.get("explicit_update_to_fia_dependency") is True
        and structure.get("external_refdata_count") == 0
        and structure.get("tensor_move_count") == 0
        and structure.get("compact_attention_and_ticket_outputs") is True
        and structure.get("full_kv_graph_output") is False
        and structure.get("data_input_abi_pass") is True
    )
    graph_pass = None
    if args.route == "combined":
        graph_pass = (
            graph.get("pass") is True
            and graph.get("device_placed_inputs_ready") is True
            and graph.get("abi_input_count") == 6
            and graph.get("metadata_input_index") == 2
            and graph.get("query_input_index") == 3
            and graph.get("attention_exact") is True
            and graph.get("ticket_exact") is True
            and graph.get("host_cache_input_bytes") == 0
            and graph.get("host_cache_output_bytes") == 0
            and all(graph_registrations[name] >= 1 for name in PATTERNS)
            and graph_transfers is not None
            and graph_transfers["no_full_cache_transfer_record"] is True
        )
    dataflow_pass = (
        dataflow.get("pass") is True
        and semantic_graph_calls_exact
        and dataflow.get("abi_input_count") == 6
        and dataflow.get("metadata_input_index") == 2
        and dataflow.get("query_input_index") == 3
        and dataflow.get("allocation_count") == 1
        and dataflow.get("flowmsg_identity_stable") is True
        and dataflow.get("buffer_address_stable") is True
        and dataflow.get("full_cache_exact_after_each_call") is True
        and dataflow.get("attention_observed_update_each_call") is True
        and dataflow.get("device_owned_cache_bytes") == 3 * 1024 * 1024
        and dataflow.get("host_cache_input_bytes") == 0
        and dataflow.get("host_cache_output_bytes") == 0
        and dataflow.get("raw_device_address_abi_used") is False
        and dataflow.get("external_refdata_used") is False
        and target_tasks_registered
        and dataflow_transfers["no_full_cache_transfer_record"] is True
    )
    ordinary_graph_required = args.route == "combined"
    result = {
        "gate": "V2-KV-ATTENTION",
        "pass": structure_pass
        and dataflow_pass
        and (graph_pass is True if ordinary_graph_required else True),
        "route": args.route,
        "structure_pass": structure_pass,
        "ordinary_graph_evaluated": ordinary_graph_required,
        "ordinary_graph_pass": graph_pass,
        "graphpp_functionpp_pass": dataflow_pass,
        "explicit_update_to_fia_dependency": structure.get(
            "explicit_update_to_fia_dependency"
        ),
        "shared_update_and_fia_kv_inputs": structure.get(
            "shared_update_and_fia_kv_inputs"
        ),
        "ordinary_graph_registration_counts": graph_registrations,
        "graphpp_registration_counts": dataflow_registrations,
        "graphpp_target_tasks_registered": target_tasks_registered,
        "semantic_graph_calls_exact": semantic_graph_calls_exact,
        "ordinary_graph_transfer_audit": graph_transfers,
        "graphpp_transfer_audit": dataflow_transfers,
        "device_owned_allocation_count": dataflow.get("allocation_count"),
        "device_owned_graph_call_count": dataflow.get("graph_call_count"),
        "device_owned_cache_bytes": dataflow.get("device_owned_cache_bytes"),
        "exact_full_cache_after_each_call": dataflow.get(
            "full_cache_exact_after_each_call"
        ),
        "attention_observed_update_each_call": dataflow.get(
            "attention_observed_update_each_call"
        ),
        "host_cache_input_bytes": dataflow.get("host_cache_input_bytes"),
        "host_cache_output_bytes": dataflow.get("host_cache_output_bytes"),
        "external_refdata_count": structure.get("external_refdata_count"),
        "tensor_move_count": structure.get("tensor_move_count"),
        "raw_device_address_abi_used": dataflow.get("raw_device_address_abi_used"),
        "claim_boundary": (
            "FunctionPp-owned PA-NZ update-to-FIA component gate only; "
            "launch logs prove task registration, while controller summaries "
            "prove the two semantic GraphPp calls; "
            "no full Decoder or P5 qualification claim; absence in bounded "
            "runtime transfer logs is not a universal zero-copy proof."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("V2_KV_ATTENTION_VERIFY " + json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
