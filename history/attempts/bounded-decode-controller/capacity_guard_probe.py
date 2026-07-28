#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import dataflow as df
import numpy as np

from bounded_decode_benchmark import build_device_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--air-path", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--graph-config", type=Path, required=True)
    parser.add_argument("--func-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = np.load(args.reference)
    initial = (
        reference["input_hidden_table"],
        reference["input_position"],
        reference["input_key_cache"],
        reference["input_value_cache"],
    )
    invalid_control = np.asarray(
        [8, 151643, 0, 4, 7, 13], dtype=np.int32
    )
    options = {
        "ge.exec.deviceId": "0",
        "ge.exec.logicalDeviceClusterDeployMode": "SINGLE",
        "ge.exec.logicalDeviceId": "[0:0]",
    }
    ret_code = None
    output_count = None
    error = None
    df.init(options)
    try:
        graph, flow_inputs = build_device_graph(args)
        graph.feed_data(
            dict(zip(flow_inputs, (*initial, invalid_control), strict=True))
        )
        try:
            outputs, _, ret_code = graph.fetch_data(timeout=300000)
            output_count = len(outputs)
        except Exception as exc:  # DataFlow versions differ on error delivery.
            error = f"{type(exc).__name__}: {exc}"
    finally:
        df.finalize()

    rejected = bool(error is not None or (ret_code != 0 and output_count == 0))
    result = {
        "schema_version": 1,
        "requested_max_steps": 8,
        "initial_position": int(initial[1].reshape(-1)[0]),
        "hidden_rows": int(initial[0].shape[0]),
        "kv_slots": int(initial[2].shape[2]),
        "ret_code": ret_code,
        "output_count": output_count,
        "exception": error,
        "capacity_guard_rejected": rejected,
        "expected": (
            "Reject before RunFlowModel because initial_position + max_steps "
            "exceeds the four staged hidden rows."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print("CAPACITY_GUARD " + json.dumps(result, ensure_ascii=True), flush=True)
    if not rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
