#!/usr/bin/env python3
"""Verify the Device-owned state lifetime and exact update probe."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LAUNCH_PATTERN = re.compile(
    r"LaunchKernel: kernel info.*kernel_name=te_devicekvslotupdate_", re.IGNORECASE
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def launch_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return len(
        LAUNCH_PATTERN.findall(path.read_text(encoding="utf-8", errors="replace"))
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
    graph_launches = launch_count(args.graph_log)
    dataflow_launches = launch_count(args.dataflow_log)
    first = dataflow.get("first_summary", [])
    second = dataflow.get("second_summary", [])
    exact_kernel_reports = (
        len(first) == 22
        and len(second) == 22
        and first[0] == 1
        and first[7] == 1
        and first[11:13] == [0, 0x3F80]
        and second[0] == 2
        and second[7] == 1
        and second[11:13] == [0x3F80, 0x4000]
    )
    graph_pass = (
        graph.get("pass") is True
        and graph.get("report_exact") is True
        and graph.get("full_cache_output_bytes") == 0
        and graph_launches == 1
    )
    dataflow_pass = (
        dataflow.get("pass") is True
        and dataflow.get("exact_two_updates") is True
        and dataflow.get("allocation_count") == 1
        and dataflow.get("graph_call_count") == 2
        and dataflow.get("buffer_address_stable") is True
        and dataflow.get("flowmsg_identity_stable") is True
        and dataflow.get("cross_call_checksum_continuity") is True
        and dataflow.get("host_cache_input_bytes") == 0
        and dataflow.get("host_cache_output_bytes") == 0
        and dataflow.get("raw_device_address_abi_used") is False
        and dataflow.get("external_refdata_used") is False
        and exact_kernel_reports
        and dataflow_launches >= 1
    )
    structure_pass = (
        structure.get("pass") is True
        and structure.get("device_kv_slot_update_count") == 1
        and structure.get("update_input_ops") == ["Data", "Data"]
        and structure.get("external_refdata_count") == 0
        and structure.get("report_only_graph_output") is True
        and structure.get("full_cache_graph_output") is False
    )
    result = {
        "gate": "V2-DEVICE-KV-UPDATE-LIFETIME",
        "pass": structure_pass and graph_pass and dataflow_pass,
        "structure_pass": structure_pass,
        "ordinary_graph_pass": graph_pass,
        "graphpp_functionpp_pass": dataflow_pass,
        "ordinary_graph_kernel_launch_count": graph_launches,
        "graphpp_kernel_launch_record_count": dataflow_launches,
        "graphpp_exact_kernel_report_count": 2 if exact_kernel_reports else 0,
        "graphpp_execution_count_source": "exact sequenced AICore reports",
        "device_owned_allocation_count": dataflow.get("allocation_count"),
        "device_owned_graph_call_count": dataflow.get("graph_call_count"),
        "same_flowmsg_across_calls": dataflow.get("flowmsg_identity_stable"),
        "same_buffer_address_across_calls": dataflow.get("buffer_address_stable"),
        "exact_cross_call_update": dataflow.get("exact_two_updates"),
        "cross_call_checksum_continuity": dataflow.get(
            "cross_call_checksum_continuity"
        ),
        "host_cache_input_bytes": dataflow.get("host_cache_input_bytes"),
        "host_cache_output_bytes": dataflow.get("host_cache_output_bytes"),
        "external_refdata_count": structure.get("external_refdata_count"),
        "raw_device_address_abi_used": dataflow.get("raw_device_address_abi_used"),
        "claim_boundary": (
            "One 8 KiB synthetic FunctionPp-owned Device buffer updated twice; "
            "no target KV layout, full Decoder, zero Device-copy, or P5 claim."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("V2_DEVICE_KV_UPDATE_VERIFY " + json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
