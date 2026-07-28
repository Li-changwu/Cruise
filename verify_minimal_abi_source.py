#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vllm_ascend_resident_epoch.abi import (
    ABI_BYTES,
    KV_ROUND_TRIP_BYTES,
    NEW_TOTAL_BYTES,
    OLD_TOTAL_BYTES,
    TOTAL_REDUCTION_BYTES,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(source: Path, baseline_source: Path | None) -> dict[str, Any]:
    source = source.resolve(strict=True)
    controller = (source / "controller/g4c_b4_resident_epoch.cpp").read_text(
        encoding="utf-8"
    )
    old_controller_path = source / "controller-old/g4c_b4_resident_epoch.cpp"
    old_controller = old_controller_path.read_text(encoding="utf-8")
    bridge = (source / "native/resident_epoch_bridge.cpp").read_text(
        encoding="utf-8"
    )
    old_bridge = (source / "native/resident_epoch_bridge_old.cpp").read_text(
        encoding="utf-8"
    )
    new_config = json.loads(
        (source / "config/resident_epoch_func.json").read_text(encoding="utf-8")
    )
    old_config = json.loads(
        (source / "config/resident_epoch_func_old.json").read_text(
            encoding="utf-8"
        )
    )
    graph_config = json.loads(
        (source / "config/graph_config.json").read_text(encoding="utf-8")
    )
    trace_source = (source / "native/rt_memcpy_trace.cpp").read_text(
        encoding="utf-8"
    )
    sidecar = (
        source / "src/vllm_ascend_resident_epoch/sidecar_backend.py"
    ).read_text(encoding="utf-8")
    trace_summary = (source / "summarize_runtime_memcpy.py").read_text(
        encoding="utf-8"
    )

    checks = {
        "new_function_config_8_in_2_out": new_config.get("input_num") == 8
        and new_config.get("output_num") == 2,
        "old_function_config_10_in_10_out": old_config.get("input_num") == 10
        and old_config.get("output_num") == 10,
        "decoder_internal_abi_unchanged": len(
            graph_config.get("inputs_tensor_desc", [])
        )
        == 9,
        "new_udf_has_no_host_kv_inputs": "kKeyCacheInput" not in controller
        and "kValueCacheInput" not in controller,
        "new_udf_has_no_logits_history_output": "logits_history" not in controller,
        "new_udf_allocates_resident_kv": "Failed to allocate resident Paged-KV state"
        in controller
        and "std::memset(resident_key_->GetTensor()->GetData(), 0" in controller
        and "std::memset(resident_value_->GetTensor()->GetData(), 0" in controller,
        "new_udf_8_in_2_out": "constexpr size_t kInputCount = 8;" in controller
        and "constexpr size_t kOutputCount = 2;" in controller,
        "new_bridge_8_in_2_out": 'FlowNode("g4c_b4_resident_epoch_node", 8, 2)'
        in bridge
        and "std::array<InputSpec, 7>" in bridge
        and "outputs.size() != 2" in bridge,
        "old_bridge_10_in_10_out": 'FlowNode("g4c_b4_resident_epoch_node", 10, 10)'
        in old_bridge
        and "std::array<InputSpec, 9>" in old_bridge
        and "outputs.size() != 10" in old_bridge,
        "new_declared_bytes_match": "kDeclaredInputBytes = 260;" in bridge
        and "kDeclaredOutputBytes = 368;" in bridge,
        "old_declared_bytes_match": "kDeclaredInputBytes = 58720516;"
        in old_bridge
        and "kDeclaredOutputBytes = 78184928;" in old_bridge,
        "logical_byte_ledger": ABI_BYTES
        == {
            "old": {"input": 58_720_516, "output": 78_184_928},
            "new": {"input": 260, "output": 368},
        }
        and OLD_TOTAL_BYTES == 136_905_444
        and NEW_TOTAL_BYTES == 628
        and TOTAL_REDUCTION_BYTES == 136_904_816
        and KV_ROUND_TRIP_BYTES == 117_440_512,
        "transfer_trace_is_measured_not_declared": (
            "dlsym(RTLD_NEXT, name)" in trace_source
            and 'Resolve<Memcpy>("rtMemcpy")' in trace_source
            and 'Resolve<MemcpyAsync>("rtMemcpyAsync")' in trace_source
            and 'Resolve<Memcpy>("rtMemcpyEx")' in trace_source
            and 'Resolve<RtsMemcpy>("rtsMemcpy")' in trace_source
            and "FeedDataFlowGraphTensor" in trace_source
            and "FetchDataFlowGraphTensor" in trace_source
            and "tensor.GetSize()" in trace_source
            and "CLOCK_REALTIME" in trace_source
            and "ASCEND_RT_MEMCPY_TRACE_PATH" in trace_source
            and "LD_PRELOAD" in sidecar
            and '"logical_abi_bytes_used_as_transfer": False' in trace_summary
            and '"dataflow_tensor_payload_is_physical_link_bytes": False'
            in trace_summary
            and '"runtime_memcpy_status"' in trace_summary
            and '"observed_zero"' in trace_summary
        ),
    }
    hashes = {
        "new_controller": _sha256(source / "controller/g4c_b4_resident_epoch.cpp"),
        "old_controller": _sha256(old_controller_path),
        "new_bridge": _sha256(source / "native/resident_epoch_bridge.cpp"),
        "old_bridge": _sha256(source / "native/resident_epoch_bridge_old.cpp"),
        "graph_config": _sha256(source / "config/graph_config.json"),
        "runtime_memcpy_trace": _sha256(source / "native/rt_memcpy_trace.cpp"),
        "transfer_trace_summarizer": _sha256(
            source / "summarize_runtime_memcpy.py"
        ),
    }
    if baseline_source is not None:
        baseline_source = baseline_source.resolve(strict=True)
        checks["old_controller_matches_attempt73"] = hashes["old_controller"] == _sha256(
            baseline_source / "controller/g4c_b4_resident_epoch.cpp"
        )
        checks["graph_config_matches_attempt73"] = hashes["graph_config"] == _sha256(
            baseline_source / "config/graph_config.json"
        )

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "pass": not failed,
        "checks": checks,
        "failed_checks": failed,
        "abi_bytes": {
            **ABI_BYTES,
            "old_total": OLD_TOTAL_BYTES,
            "new_total": NEW_TOTAL_BYTES,
            "total_reduction": TOTAL_REDUCTION_BYTES,
            "kv_round_trip": KV_ROUND_TRIP_BYTES,
        },
        "sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate_source(args.source, args.baseline_source)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"MINIMAL_ABI_SOURCE_INVALID: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
