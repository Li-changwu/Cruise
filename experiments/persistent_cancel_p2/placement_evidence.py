#!/usr/bin/env python3
"""Extract fail-closed P2 controller placement evidence from CANN logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FORMAT = "cruise-persistent-cancel-p2-placement-v1"
TARGET_PROCESS_POINT = "persistent_cancel_p2_pp"
MAX_EVIDENCE_LINES = 128

_DEPLOYMENT = re.compile(
    rf"Model deployment info, model_name = {TARGET_PROCESS_POINT},.*?"
    r"node_type = ([^,\s.]+)",
    re.IGNORECASE,
)
_SELECTED_RESOURCE = re.compile(
    r"select resource type(?: is)?\s*\[([^\]]+)\]", re.IGNORECASE
)
_RUNNABLE_RESOURCES = re.compile(
    rf"Get pp\[{TARGET_PROCESS_POINT}\]'s runnable resource info\[([^\]]*)\]",
    re.IGNORECASE,
)
_MODEL_DEVICE_TYPE = re.compile(
    rf"model_name = {TARGET_PROCESS_POINT},.*?device_type = (\d+)", re.IGNORECASE
)
_DEVICE_PROXY = re.compile(r"\[udf_proxy_client\.cc:", re.IGNORECASE)
_PLACEMENT_PROBE = re.compile(
    r"P2_PLACEMENT_PROBE .*feed_calls=2 commits=1 aicore_calls=1 "
    r"retired=1 quiescent=1 .*compile_status=0 fetch_status=0 "
    r"remove_status=0 finalize_status=0"
)
_FORBIDDEN = {
    "local_udfs": re.compile(r"\blocal udfs\b", re.IGNORECASE),
    "host_udf_untar": re.compile(r"\bhost udf do untar\b", re.IGNORECASE),
    "host_udf_executor": re.compile(
        r"\[udf_executor_client\.cc:[^\]]+\].*LoadProcess:Fork udf process",
        re.IGNORECASE,
    ),
}


def _log_files(roots: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        resolved = root.resolve(strict=True)
        if resolved.is_file():
            files.add(resolved)
        else:
            files.update(
                path.resolve() for path in resolved.rglob("*") if path.is_file()
            )
    return sorted(files)


def analyze_placement(log_roots: list[Path]) -> dict[str, Any]:
    files = _log_files(log_roots)
    node_types: list[str] = []
    selected_resources: list[str] = []
    runnable_resources: list[str] = []
    model_device_types: list[int] = []
    device_proxy_records = 0
    placement_probe_records = 0
    forbidden_counts = {name: 0 for name in _FORBIDDEN}
    evidence_lines: list[str] = []
    bytes_scanned = 0
    for path in files:
        bytes_scanned += path.stat().st_size
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = raw_line.rstrip("\r\n")
                matched = False
                deployment = _DEPLOYMENT.search(line)
                if deployment is not None:
                    node_types.append(deployment.group(1))
                    matched = True
                selection = _SELECTED_RESOURCE.search(line)
                if selection is not None:
                    selected_resources.append(selection.group(1))
                    matched = True
                runnable = _RUNNABLE_RESOURCES.search(line)
                if runnable is not None:
                    runnable_resources.append(runnable.group(1).strip("[]"))
                    matched = True
                device_type = _MODEL_DEVICE_TYPE.search(line)
                if device_type is not None:
                    model_device_types.append(int(device_type.group(1)))
                    matched = True
                if _DEVICE_PROXY.search(line) is not None:
                    device_proxy_records += 1
                    matched = True
                if _PLACEMENT_PROBE.search(line) is not None:
                    placement_probe_records += 1
                    matched = True
                for name, pattern in _FORBIDDEN.items():
                    if pattern.search(line) is not None:
                        forbidden_counts[name] += 1
                        matched = True
                if matched and len(evidence_lines) < MAX_EVIDENCE_LINES:
                    evidence_lines.append(f"{path}:{line_number}:{line}")

    errors: list[str] = []
    normalized = [value.strip().lower() for value in selected_resources]
    if not node_types:
        errors.append(f"no deployment record for {TARGET_PROCESS_POINT}")
    if "ascend" not in normalized:
        errors.append(
            f"Ascend resource selection is not proven: {selected_resources}"
        )
    if not model_device_types or any(value != 0 for value in model_device_types):
        errors.append(
            f"controller Device model type is not proven: {model_device_types}"
        )
    if device_proxy_records == 0:
        errors.append("Device UDF proxy deployment is not proven")
    if placement_probe_records != 1:
        errors.append(
            f"successful P2 placement probes={placement_probe_records}, expected 1"
        )
    present_forbidden = {
        name: count for name, count in forbidden_counts.items() if count > 0
    }
    if present_forbidden:
        errors.append(f"Host-local deployment evidence is present: {present_forbidden}")
    return {
        "format": FORMAT,
        "pass": not errors,
        "target_process_point": TARGET_PROCESS_POINT,
        "errors": errors,
        "deployment_node_types": node_types,
        "selected_resource_types": selected_resources,
        "runnable_resource_info": runnable_resources,
        "model_device_types": model_device_types,
        "device_proxy_records": device_proxy_records,
        "placement_probe_records": placement_probe_records,
        "forbidden_evidence_counts": forbidden_counts,
        "files_scanned": [str(path) for path in files],
        "bytes_scanned": bytes_scanned,
        "evidence_lines": evidence_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_placement(args.log_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.extract_output.write_text(
        "\n".join(result["evidence_lines"])
        + ("\n" if result["evidence_lines"] else ""),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
