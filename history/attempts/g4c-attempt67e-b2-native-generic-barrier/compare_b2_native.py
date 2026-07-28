#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3
NUM_LAYERS = 28
BATCH_SIZE = 2
PHYSICAL_BLOCKS = 4
BLOCK_SIZE = 128
NUM_KV_HEADS = 4
HEAD_DIM = 128
VOCAB_SIZE = 152064
CACHE_SHAPE = (NUM_LAYERS, PHYSICAL_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
LOGITS_SHAPE = (BATCH_SIZE, 1, VOCAB_SIZE)
CASES = (
    "both-active-heterogeneous",
    "active-plus-empty",
    "finished-plus-active",
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


def bf16_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual_fp32 = (actual.astype(np.uint32) << np.uint32(16)).view(np.float32)
    expected_fp32 = (expected.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return metrics(actual_fp32, expected_fp32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    archive = {}
    with np.load(args.reference) as reference:
        for case_index, case_name in enumerate(CASES):
            prefix = f"case{case_index}"
            logits = np.fromfile(
                args.native_dir / f"{prefix}_logits.bin", dtype=np.float32
            ).reshape(LOGITS_SHAPE)
            key = np.fromfile(
                args.native_dir / f"{prefix}_key_cache.bin", dtype=np.uint16
            ).reshape(CACHE_SHAPE)
            value = np.fromfile(
                args.native_dir / f"{prefix}_value_cache.bin", dtype=np.uint16
            ).reshape(CACHE_SHAPE)
            position = np.fromfile(
                args.native_dir / f"{prefix}_next_position.bin", dtype=np.int64
            ).reshape(BATCH_SIZE)
            expected_logits = reference[f"{prefix}_b2_logits"]
            expected_key = reference[f"{prefix}_b2_key_cache_bits"]
            expected_value = reference[f"{prefix}_b2_value_cache_bits"]
            expected_position = reference[f"{prefix}_b2_next_position"]
            input_key = reference[f"{prefix}_input_key_cache_bits"]
            input_value = reference[f"{prefix}_input_value_cache_bits"]
            active = reference[f"{prefix}_active_mask"]
            slots = reference[f"{prefix}_slot_mapping"]
            mutable = np.zeros(PHYSICAL_BLOCKS * BLOCK_SIZE, dtype=bool)
            for request in range(BATCH_SIZE):
                if active[request] != 0:
                    mutable[int(slots[request])] = True
            flat_key = key.reshape(NUM_LAYERS, PHYSICAL_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
            flat_value = value.reshape(flat_key.shape)
            flat_input_key = input_key.reshape(flat_key.shape)
            flat_input_value = input_value.reshape(flat_key.shape)
            inactive_cache_exact = []
            active_greedy_equal = []
            active_logits_finite = []
            for request in range(BATCH_SIZE):
                if active[request] == 0:
                    begin = request * 2
                    end = begin + 2
                    inactive_cache_exact.append(
                        bool(
                            np.array_equal(key[:, begin:end], input_key[:, begin:end])
                            and np.array_equal(value[:, begin:end], input_value[:, begin:end])
                        )
                    )
                else:
                    active_greedy_equal.append(
                        int(np.argmax(logits[request].reshape(-1)))
                        == int(np.argmax(expected_logits[request].reshape(-1)))
                    )
                    active_logits_finite.append(bool(np.all(np.isfinite(logits[request]))))
            logits_metric = metrics(logits, expected_logits)
            key_metric = bf16_metrics(key, expected_key)
            value_metric = bf16_metrics(value, expected_value)
            item = {
                "case": case_name,
                "logits_vs_b2_eager": logits_metric,
                "key_cache_vs_b2_eager": key_metric,
                "value_cache_vs_b2_eager": value_metric,
                "next_position_equal": bool(np.array_equal(position, expected_position)),
                "next_position_actual": position.tolist(),
                "next_position_expected": expected_position.tolist(),
                "active_greedy_equal": all(active_greedy_equal),
                "active_logits_finite": all(active_logits_finite),
                "unaddressed_key_elementwise_exact": bool(
                    np.array_equal(flat_key[:, ~mutable], flat_input_key[:, ~mutable])
                ),
                "unaddressed_value_elementwise_exact": bool(
                    np.array_equal(flat_value[:, ~mutable], flat_input_value[:, ~mutable])
                ),
                "inactive_request_cache_elementwise_exact": all(inactive_cache_exact),
            }
            item["pass"] = bool(
                logits_metric["all_within_tolerance"]
                and key_metric["all_within_tolerance"]
                and value_metric["all_within_tolerance"]
                and item["next_position_equal"]
                and item["active_greedy_equal"]
                and item["active_logits_finite"]
                and item["unaddressed_key_elementwise_exact"]
                and item["unaddressed_value_elementwise_exact"]
                and item["inactive_request_cache_elementwise_exact"]
            )
            cases.append(item)
            archive[f"{prefix}_logits"] = logits
            archive[f"{prefix}_key_cache_bits"] = key
            archive[f"{prefix}_value_cache_bits"] = value
            archive[f"{prefix}_next_position"] = position
    np.savez(args.output_npz, **archive)
    result = {
        "gate": "G4c Attempt 67e B=2 native GE with generic barrier versus batched eager",
        "pass": all(case["pass"] for case in cases),
        "execution_success": True,
        "rtol": RTOL,
        "atol": ATOL,
        "reference_sha256": sha256(args.reference),
        "native_output_sha256": sha256(args.output_npz),
        "cases": cases,
        "claim_boundary": (
            "This closes B=2 single-step native GE semantics only. Recurrent "
            "Host/Device epochs, independent EOS, B=4, performance, recovery, "
            "and vLLM integration remain open."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
