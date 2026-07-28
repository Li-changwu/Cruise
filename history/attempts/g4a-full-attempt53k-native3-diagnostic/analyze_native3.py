#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3
NUM_LAYERS = 28
PHYSICAL_SLOTS = 256
NUM_KV_HEADS = 4
HEAD_DIM = 128
CACHE_SHAPE = (NUM_LAYERS, 2, 128, NUM_KV_HEADS, HEAD_DIM)
LOGITS_SHAPE = (1, 1, 152064)


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
    exact = actual == expected
    close = np.isclose(actual_fp32, expected_fp32, rtol=RTOL, atol=ATOL)
    absolute = np.abs(actual_fp32 - expected_fp32)
    return {
        "all_exact": bool(np.all(exact)),
        "all_within_tolerance": bool(np.all(close)),
        "exact_mismatch_count": int(exact.size - np.count_nonzero(exact)),
        "tolerance_failure_count": int(close.size - np.count_nonzero(close)),
        "max_abs_error": float(np.max(absolute)),
        "mean_abs_error": float(np.mean(absolute)),
    }


def bf16_metrics(actual_bits: np.ndarray, expected_bits: np.ndarray) -> dict:
    return metrics(bf16_to_float32(actual_bits), bf16_to_float32(expected_bits))


def topk(logits: np.ndarray, count: int = 10) -> list[dict]:
    flat = logits.reshape(-1)
    indices = np.argsort(flat)[-count:][::-1]
    return [{"token": int(index), "logit": float(flat[index])} for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initial-input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial_key = np.fromfile(
        args.initial_input_dir / "initial_key_cache.bin", dtype=np.uint16
    ).reshape(CACHE_SHAPE)
    initial_value = np.fromfile(
        args.initial_input_dir / "initial_value_cache.bin", dtype=np.uint16
    ).reshape(CACHE_SHAPE)

    result = {
        "gate": "G4a Attempt 53k native3 layer localization",
        "rtol": RTOL,
        "atol": ATOL,
        "reference_sha256": sha256(args.reference),
        "native_files": {},
        "input_exact": {},
        "steps": [],
    }

    with np.load(args.reference) as reference:
        result["input_exact"] = {
            "key_cache": bool(
                np.array_equal(initial_key, reference["input_key_cache_bits"])
            ),
            "value_cache": bool(
                np.array_equal(initial_value, reference["input_value_cache_bits"])
            ),
            "block_table": bool(
                np.array_equal(
                    np.fromfile(
                        args.initial_input_dir / "block_table.bin", dtype=np.int32
                    ).reshape(reference["block_table"].shape),
                    reference["block_table"],
                )
            ),
            "tiling": bool(
                np.array_equal(
                    np.fromfile(
                        args.initial_input_dir / "tiling.bin", dtype=np.uint8
                    ),
                    reference["tiling"],
                )
            ),
        }

        previous_key = initial_key
        previous_value = initial_value
        for step in range(1, 5):
            paths = {
                name: args.native_dir / f"step{step}_{name}.bin"
                for name in ("logits", "key_cache", "value_cache", "next_position")
            }
            for name, path in paths.items():
                result["native_files"][f"step{step}_{name}"] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }

            logits = np.fromfile(paths["logits"], dtype=np.float32).reshape(LOGITS_SHAPE)
            key = np.fromfile(paths["key_cache"], dtype=np.uint16).reshape(CACHE_SHAPE)
            value = np.fromfile(paths["value_cache"], dtype=np.uint16).reshape(CACHE_SHAPE)
            expected_logits = reference[f"step{step}_logits"]
            expected_key = reference[f"step{step}_key_cache_bits"]
            expected_value = reference[f"step{step}_value_cache_bits"]
            addressed = 128 + step - 1
            flat_key = key.reshape(NUM_LAYERS, PHYSICAL_SLOTS, NUM_KV_HEADS, HEAD_DIM)
            flat_value = value.reshape(NUM_LAYERS, PHYSICAL_SLOTS, NUM_KV_HEADS, HEAD_DIM)
            expected_flat_key = expected_key.reshape(
                NUM_LAYERS, PHYSICAL_SLOTS, NUM_KV_HEADS, HEAD_DIM
            )
            expected_flat_value = expected_value.reshape(
                NUM_LAYERS, PHYSICAL_SLOTS, NUM_KV_HEADS, HEAD_DIM
            )
            previous_flat_key = previous_key.reshape(
                NUM_LAYERS, PHYSICAL_SLOTS, NUM_KV_HEADS, HEAD_DIM
            )
            previous_flat_value = previous_value.reshape(
                NUM_LAYERS, PHYSICAL_SLOTS, NUM_KV_HEADS, HEAD_DIM
            )

            layers = []
            for layer in range(NUM_LAYERS):
                key_metric = bf16_metrics(
                    flat_key[layer, addressed], expected_flat_key[layer, addressed]
                )
                value_metric = bf16_metrics(
                    flat_value[layer, addressed], expected_flat_value[layer, addressed]
                )
                layers.append(
                    {
                        "layer": layer,
                        "key": key_metric,
                        "value": value_metric,
                        "pass": key_metric["all_within_tolerance"]
                        and value_metric["all_within_tolerance"],
                    }
                )

            unaddressed = np.ones(PHYSICAL_SLOTS, dtype=bool)
            unaddressed[addressed] = False
            first_failure = next(
                (item["layer"] for item in layers if not item["pass"]), None
            )
            logits_metric = metrics(logits, expected_logits)
            result["steps"].append(
                {
                    "step": step,
                    "addressed_slot": addressed,
                    "first_failing_layer": first_failure,
                    "passing_layer_count": sum(item["pass"] for item in layers),
                    "layers": layers,
                    "logits": logits_metric,
                    "native_top10": topk(logits),
                    "eager_top10": topk(expected_logits),
                    "unaddressed_key_exact": bool(
                        np.array_equal(
                            flat_key[:, unaddressed], previous_flat_key[:, unaddressed]
                        )
                    ),
                    "unaddressed_value_exact": bool(
                        np.array_equal(
                            flat_value[:, unaddressed], previous_flat_value[:, unaddressed]
                        )
                    ),
                }
            )
            previous_key = key
            previous_value = value

    first_failures = [item["first_failing_layer"] for item in result["steps"]]
    result["diagnosis"] = {
        "first_failing_layers": first_failures,
        "all_inputs_exact": all(result["input_exact"].values()),
        "boundary": (
            "Layer L written K/V is computed before layer L attention. A layer-0 "
            "failure localizes the discrepancy to embedding/RMSNorm/QKV/RoPE; a "
            "later first failure also includes the preceding layer's attention/MLP."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["diagnosis"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
