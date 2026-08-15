#!/usr/bin/env python3
"""Verify the target-layout Paged-KV update/read ordering probe."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


UPDATE_PATTERN = re.compile(
    r"LaunchKernel: kernel info.*kernel_name=te_devicepagedkvupdate_",
    re.IGNORECASE,
)
READ_PATTERN = re.compile(
    r"LaunchKernel: kernel info.*kernel_name=te_devicepagedkvread_",
    re.IGNORECASE,
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def launch_counts(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(UPDATE_PATTERN.findall(text)), len(READ_PATTERN.findall(text))


def exact_summaries(dataflow: dict) -> bool:
    first = dataflow.get("first_summary", [])
    second = dataflow.get("second_summary", [])
    return (
        len(first) == 24
        and len(second) == 24
        and first[0] == 1
        and second[0] == 2
        and first[1] == 0
        and second[1] == 0
        and first[7:9] == [1, 1]
        and second[7:9] == [1, 1]
        and first[18] == 0
        and second[18] == 0
        and first[19] == first[20] == first[23]
        and second[19] == second[20] == second[23]
        and first[21:23] == [0, 0]
        and second[21:23] == [0, 0]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--graph-result", type=Path, required=True)
    parser.add_argument("--graph-log", type=Path, required=True)
    parser.add_argument("--dataflow-result", type=Path, required=True)
    parser.add_argument("--dataflow-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    structure = read_json(args.structure)
    graph = read_json(args.graph_result)
    dataflow = read_json(args.dataflow_result)
    graph_update_launches, graph_read_launches = launch_counts(args.graph_log)
    dataflow_update_launches, dataflow_read_launches = launch_counts(
        args.dataflow_log
    )
    summaries_exact = exact_summaries(dataflow)

    structure_pass = (
        structure.get("pass") is True
        and structure.get("device_paged_kv_update_count") == 1
        and structure.get("device_paged_kv_read_count") == 1
        and structure.get("shared_state_input") is True
        and structure.get("explicit_update_to_reader_dependency") is True
        and structure.get("external_refdata_count") == 0
        and structure.get("tensor_move_count") == 0
        and structure.get("report_only_graph_output") is True
        and structure.get("full_state_graph_output") is False
    )
    graph_pass = (
        graph.get("pass") is True
        and graph.get("report_exact") is True
        and graph.get("explicit_dependency_observed") is True
        and graph.get("full_state_output_bytes") == 0
        and graph_update_launches == 1
        and graph_read_launches == 1
    )
    dataflow_pass = (
        dataflow.get("pass") is True
        and dataflow.get("exact_two_ordered_updates") is True
        and dataflow.get("allocation_count") == 1
        and dataflow.get("graph_call_count") == 2
        and dataflow.get("buffer_address_stable") is True
        and dataflow.get("flowmsg_identity_stable") is True
        and dataflow.get("cross_call_checksum_continuity") is True
        and dataflow.get("full_state_exact_after_each_call") is True
        and dataflow.get("explicit_dependency_observed_each_call") is True
        and dataflow.get("device_owned_state_bytes") == 3 * 1024 * 1024
        and dataflow.get("host_cache_input_bytes") == 0
        and dataflow.get("host_cache_output_bytes") == 0
        and dataflow.get("raw_device_address_abi_used") is False
        and dataflow.get("external_refdata_used") is False
        and summaries_exact
        and dataflow_update_launches >= 1
        and dataflow_read_launches >= 1
    )
    result = {
        "gate": "V2-KV-ORDER",
        "pass": structure_pass and graph_pass and dataflow_pass,
        "structure_pass": structure_pass,
        "ordinary_graph_pass": graph_pass,
        "graphpp_functionpp_pass": dataflow_pass,
        "explicit_update_to_reader_dependency": structure.get(
            "explicit_update_to_reader_dependency"
        ),
        "ordinary_graph_update_launch_count": graph_update_launches,
        "ordinary_graph_reader_launch_count": graph_read_launches,
        "graphpp_update_launch_record_count": dataflow_update_launches,
        "graphpp_reader_launch_record_count": dataflow_read_launches,
        "graphpp_exact_ordered_report_count": 2 if summaries_exact else 0,
        "graphpp_execution_count_source": "exact sequence-dependent Device reports",
        "device_owned_allocation_count": dataflow.get("allocation_count"),
        "device_owned_graph_call_count": dataflow.get("graph_call_count"),
        "device_owned_state_bytes": dataflow.get("device_owned_state_bytes"),
        "same_flowmsg_across_calls": dataflow.get("flowmsg_identity_stable"),
        "same_buffer_address_across_calls": dataflow.get("buffer_address_stable"),
        "exact_full_state_after_each_call": dataflow.get(
            "full_state_exact_after_each_call"
        ),
        "cross_call_checksum_continuity": dataflow.get(
            "cross_call_checksum_continuity"
        ),
        "host_cache_input_bytes": dataflow.get("host_cache_input_bytes"),
        "host_cache_output_bytes": dataflow.get("host_cache_output_bytes"),
        "external_refdata_count": structure.get("external_refdata_count"),
        "raw_device_address_abi_used": dataflow.get("raw_device_address_abi_used"),
        "claim_boundary": (
            "B4/K384 Paged-KV layout and explicit update-to-reader ordering only; "
            "no attention, full Decoder, internal Device-copy, or P5 claim."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("V2_KV_ORDER_VERIFY " + json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
