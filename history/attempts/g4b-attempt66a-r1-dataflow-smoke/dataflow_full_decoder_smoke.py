#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import dataflow as df
import dataflow.data_type as df_dt
import dataflow.data_wrapper as data_wrapper
import dataflow.dataflow as df_impl
import dataflow.dflow_wrapper as dwrapper
import numpy as np


RTOL = 5e-3
ATOL = 5e-3
CACHE_SHAPE = (28, 2, 128, 4, 128)
LOGITS_SHAPE = (1, 1, 152064)


def enable_bf16_numpy_bridge() -> None:
    """Treat uint16 arrays as BF16 bit containers inside this process."""
    bf16_dtype = np.dtype(np.uint16)
    bf16_enum = data_wrapper.DataType.DT_BF16
    df_dt._dwrapper_dtype_to_python_dtype_str[bf16_enum] = "DT_BF16"
    df_dt._dwrapper_dtype_to_python_dtype[bf16_enum] = df_dt.DT_BF16
    df_dt._dflow_dtype_to_np_dtype[df_dt.DT_BF16] = np.uint16
    df_dt._np_dtype_to_dflow_dtype[bf16_dtype] = df_dt.DT_BF16
    dwrapper.init_datatype_manager(
        {
            data_wrapper.DataType.DT_FLOAT: np.array([], np.float32),
            data_wrapper.DataType.DT_FLOAT16: np.array([], np.float16),
            data_wrapper.DataType.DT_BF16: np.array([], np.uint16),
            data_wrapper.DataType.DT_INT8: np.array([], np.int8),
            data_wrapper.DataType.DT_INT16: np.array([], np.int16),
            data_wrapper.DataType.DT_UINT8: np.array([], np.uint8),
            data_wrapper.DataType.DT_INT32: np.array([], np.int32),
            data_wrapper.DataType.DT_INT64: np.array([], np.int64),
            data_wrapper.DataType.DT_UINT32: np.array([], np.uint32),
            data_wrapper.DataType.DT_UINT64: np.array([], np.uint64),
            data_wrapper.DataType.DT_BOOL: np.array([], np.bool_),
            data_wrapper.DataType.DT_DOUBLE: np.array([], np.float64),
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual_fp32 = actual.astype(np.float32)
    expected_fp32 = expected.astype(np.float32)
    exact = np.equal(actual, expected)
    close = np.isclose(actual_fp32, expected_fp32, rtol=RTOL, atol=ATOL)
    return {
        "all_exact": bool(np.all(exact)),
        "all_within_tolerance": bool(np.all(close)),
        "max_abs_error": float(np.max(np.abs(actual_fp32 - expected_fp32))),
        "exact_mismatch_count": int(exact.size - np.count_nonzero(exact)),
        "tolerance_failure_count": int(close.size - np.count_nonzero(close)),
    }


def bf16_metrics(actual_bits: np.ndarray, expected_bits: np.ndarray) -> dict:
    actual = (actual_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    expected = (expected_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return metrics(actual, expected)


def as_bfloat16(bits: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bits.astype(np.uint16, copy=False))


def unpack_fetch(fetch_result) -> list[np.ndarray]:
    if not isinstance(fetch_result, tuple) or len(fetch_result) != 3:
        raise RuntimeError(f"unexpected fetch result: {type(fetch_result)}")
    outputs, _, ret_code = fetch_result
    if ret_code != 0 or len(outputs) != 4:
        raise RuntimeError(f"fetch failed ret_code={ret_code} outputs={len(outputs)}")
    return [np.asarray(output.numpy()).copy() for output in outputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--air", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--graph-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enable_bf16_numpy_bridge()

    with np.load(args.reference) as reference:
        inputs = [
            reference["token_ids"][:1].copy().reshape(1, 1),
            np.asarray([0], dtype=np.int64),
            np.asarray([[1]], dtype=np.int32),
            reference["block_table"].copy(),
            np.asarray([128], dtype=np.int32),
            as_bfloat16(reference["input_key_cache_bits"]),
            as_bfloat16(reference["input_value_cache_bits"]),
            reference["tiling"].copy(),
        ]
        expected_logits = reference["step1_logits"].copy()
        expected_key = reference["step1_key_cache_bits"].copy()
        expected_value = reference["step1_value_cache_bits"].copy()
        expected_position = reference["step1_next_position"].copy()

    options = {
        "ge.exec.deviceId": "0",
        "ge.exec.logicalDeviceClusterDeployMode": "SINGLE",
        "ge.exec.logicalDeviceId": "[0:0]",
        "ge.exec.precision_mode": "must_keep_origin_dtype",
    }
    df.init(options)
    try:
        flow_inputs = [df.FlowData() for _ in range(8)]
        graph_pp = df.GraphProcessPoint(
            df.Framework.MINDSPORE,
            str(args.air),
            compile_config_path=str(args.graph_config),
            name="attempt66a_full_decoder_air",
        )
        node = df.FlowNode(input_num=8, output_num=4, name="attempt66a_decoder_node")
        node.add_process_point(graph_pp)
        flow_outputs = node(*flow_inputs)
        graph = df.FlowGraph(list(flow_outputs), name="attempt66a_dataflow_smoke")
        graph.feed_data(dict(zip(flow_inputs, inputs, strict=True)))
        logits, key, value, next_position = unpack_fetch(
            graph.fetch_data(timeout=600000)
        )
    finally:
        df.finalize()

    key_bits = np.ascontiguousarray(key).view(np.uint16).reshape(CACHE_SHAPE)
    value_bits = np.ascontiguousarray(value).view(np.uint16).reshape(CACHE_SHAPE)
    logits = logits.reshape(LOGITS_SHAPE)
    next_position = next_position.astype(np.int64, copy=False).reshape(1)
    result = {
        "gate": "G4b Attempt 66a-r1 DataFlow full-decoder BF16 smoke",
        "pass": False,
        "air_sha256": sha256(args.air),
        "reference_sha256": sha256(args.reference),
        "input_dtypes": [str(value.dtype) for value in inputs],
        "output_dtypes": [str(item.dtype) for item in (logits, key, value, next_position)],
        "output_shapes": [list(item.shape) for item in (logits, key, value, next_position)],
        "logits_vs_eager": metrics(logits, expected_logits),
        "key_cache_vs_eager": bf16_metrics(key_bits, expected_key),
        "value_cache_vs_eager": bf16_metrics(value_bits, expected_value),
        "position_equal": bool(np.array_equal(next_position, expected_position)),
        "bf16_bridge": (
            "process-local bit-preserving uint16 container registered as DT_BF16; "
            "no genuine DT_UINT16 tensors are used"
        ),
        "claim_boundary": (
            "One Host DataFlow invocation only; Device UDF recurrence, sampling and EOS "
            "are not tested."
        ),
    }
    result["pass"] = bool(
        result["logits_vs_eager"]["all_within_tolerance"]
        and result["key_cache_vs_eager"]["all_within_tolerance"]
        and result["value_cache_vs_eager"]["all_within_tolerance"]
        and result["position_equal"]
    )
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
