#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3
CACHE_SHAPE = (28, 2, 128, 4, 128)
LOGITS_SHAPE = (1, 1, 152064)
LAYER_HIDDENS_SHAPE = (28, 1, 1, 3584)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


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
    return metrics(bf16_to_float32(actual_bits), bf16_to_float32(expected_bits))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initial-input-dir", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    saved = {}
    steps = []
    previous_key = np.fromfile(
        args.initial_input_dir / "initial_key_cache.bin", dtype=np.uint16
    ).reshape(CACHE_SHAPE)
    previous_value = np.fromfile(
        args.initial_input_dir / "initial_value_cache.bin", dtype=np.uint16
    ).reshape(CACHE_SHAPE)

    with np.load(args.reference) as reference:
        for step in range(1, 5):
            logits = np.fromfile(
                args.native_dir / f"step{step}_logits.bin", dtype=np.float32
            ).reshape(LOGITS_SHAPE)
            key_bits = np.fromfile(
                args.native_dir / f"step{step}_key_cache.bin", dtype=np.uint16
            ).reshape(CACHE_SHAPE)
            value_bits = np.fromfile(
                args.native_dir / f"step{step}_value_cache.bin", dtype=np.uint16
            ).reshape(CACHE_SHAPE)
            next_position = np.fromfile(
                args.native_dir / f"step{step}_next_position.bin", dtype=np.int64
            ).reshape(1)
            layer_hidden_bits = np.fromfile(
                args.native_dir / f"step{step}_layer_hiddens.bin", dtype=np.uint16
            ).reshape(LAYER_HIDDENS_SHAPE)
            expected_logits = reference[f"step{step}_logits"]
            expected_key = reference[f"step{step}_key_cache_bits"]
            expected_value = reference[f"step{step}_value_cache_bits"]
            expected_position = reference[f"step{step}_next_position"]
            expected_layer_hidden_bits = reference[f"step{step}_layer_hidden_bits"]

            addressed = 128 + step - 1
            flat_key = key_bits.reshape(28, 256, 4, 128)
            flat_value = value_bits.reshape(28, 256, 4, 128)
            flat_previous_key = previous_key.reshape(28, 256, 4, 128)
            flat_previous_value = previous_value.reshape(28, 256, 4, 128)
            unaddressed = np.ones(256, dtype=bool)
            unaddressed[addressed] = False
            key_metric = bf16_metrics(key_bits, expected_key)
            value_metric = bf16_metrics(value_bits, expected_value)
            written_key_metric = bf16_metrics(
                flat_key[:, addressed : addressed + 1],
                expected_key.reshape(28, 256, 4, 128)[:, addressed : addressed + 1],
            )
            written_value_metric = bf16_metrics(
                flat_value[:, addressed : addressed + 1],
                expected_value.reshape(28, 256, 4, 128)[:, addressed : addressed + 1],
            )
            logits_metric = metrics(logits, expected_logits)
            layer_hidden_metrics = [
                bf16_metrics(layer_hidden_bits[layer], expected_layer_hidden_bits[layer])
                for layer in range(LAYER_HIDDENS_SHAPE[0])
            ]
            item = {
                "step": step,
                "logits_vs_eager": logits_metric,
                "key_cache_vs_eager": key_metric,
                "value_cache_vs_eager": value_metric,
                "written_key_vs_eager": written_key_metric,
                "written_value_vs_eager": written_value_metric,
                "unaddressed_key_elementwise_exact": bool(
                    np.array_equal(flat_key[:, unaddressed], flat_previous_key[:, unaddressed])
                ),
                "unaddressed_value_elementwise_exact": bool(
                    np.array_equal(flat_value[:, unaddressed], flat_previous_value[:, unaddressed])
                ),
                "native_greedy": int(np.argmax(logits.reshape(-1))),
                "eager_greedy": int(np.argmax(expected_logits.reshape(-1))),
                "logits_finite": bool(np.all(np.isfinite(logits))),
                "next_position": int(next_position[0]),
                "expected_next_position": int(expected_position[0]),
                "first_layer_hidden_exact_mismatch": next(
                    (
                        layer
                        for layer, metric in enumerate(layer_hidden_metrics)
                        if not metric["all_exact"]
                    ),
                    None,
                ),
                "first_layer_hidden_tolerance_failure": next(
                    (
                        layer
                        for layer, metric in enumerate(layer_hidden_metrics)
                        if not metric["all_within_tolerance"]
                    ),
                    None,
                ),
                "layer_hiddens_vs_eager": layer_hidden_metrics,
            }
            item["greedy_equal"] = item["native_greedy"] == item["eager_greedy"]
            item["pass"] = (
                logits_metric["all_within_tolerance"]
                and key_metric["all_within_tolerance"]
                and value_metric["all_within_tolerance"]
                and written_key_metric["all_within_tolerance"]
                and written_value_metric["all_within_tolerance"]
                and item["unaddressed_key_elementwise_exact"]
                and item["unaddressed_value_elementwise_exact"]
                and item["greedy_equal"]
                and item["logits_finite"]
                and item["next_position"] == item["expected_next_position"]
            )
            steps.append(item)
            saved[f"step{step}_logits"] = logits
            saved[f"step{step}_key_cache_bits"] = key_bits
            saved[f"step{step}_value_cache_bits"] = value_bits
            saved[f"step{step}_next_position"] = next_position
            saved[f"step{step}_layer_hidden_bits"] = layer_hidden_bits
            previous_key = key_bits.copy()
            previous_value = value_bits.copy()

    np.savez(args.output_npz, **saved)
    result = {
        "gate": "G4a Attempt 64a residual-defusion diagnostic",
        "pass": all(item["pass"] for item in steps),
        "diagnostic_complete": len(steps) == 4,
        "execution_success": True,
        "rtol": RTOL,
        "atol": ATOL,
        "reference_sha256": sha256(args.reference),
        "native_output_sha256": sha256(args.output_npz),
        "steps": steps,
        "diagnosis": {
            "first_layer_hidden_exact_mismatch": [
                item["first_layer_hidden_exact_mismatch"] for item in steps
            ],
            "first_layer_hidden_tolerance_failure": [
                item["first_layer_hidden_tolerance_failure"] for item in steps
            ],
        },
        "claim_boundary": "Diagnostic only; this result cannot pass G4a.",
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["diagnostic_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
