#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3


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
    difference = np.abs(actual_fp32 - expected_fp32)
    return {
        "all_exact": bool(np.all(exact)),
        "all_within_tolerance": bool(np.all(close)),
        "max_abs_error": float(np.max(difference)),
        "mean_abs_error": float(np.mean(difference)),
        "exact_mismatch_count": int(exact.size - np.count_nonzero(exact)),
        "tolerance_failure_count": int(close.size - np.count_nonzero(close)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eager-screen", type=Path, required=True)
    parser.add_argument("--initial-input-dir", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.eager_screen.read_text(encoding="utf-8"))["output_specs"]
    initial_key = np.fromfile(
        args.initial_input_dir / "initial_key_cache.bin", dtype=np.uint16
    ).reshape(2, 128, 4, 128)
    initial_value = np.fromfile(
        args.initial_input_dir / "initial_value_cache.bin", dtype=np.uint16
    ).reshape(2, 128, 4, 128)
    previous_key, previous_value = initial_key, initial_value
    steps, saved = [], {}
    with np.load(args.reference) as reference:
        for step in range(1, 5):
            item = {"step": step, "outputs": {}}
            finite = True
            for spec in specs:
                name = spec["name"]
                shape = tuple(spec["shape"])
                if spec["abi_dtype"] == "DT_BF16":
                    raw = np.fromfile(
                        args.native_dir / f"step{step}_{name}.bin", dtype=np.uint16
                    ).reshape(shape)
                    expected = reference[f"step{step}_{name}_bits"]
                    observed = bf16_to_float32(raw)
                    expected_observed = bf16_to_float32(expected)
                    metric = metrics(observed, expected_observed)
                    finite = finite and bool(np.all(np.isfinite(observed)))
                elif spec["abi_dtype"] == "DT_INT64":
                    raw = np.fromfile(
                        args.native_dir / f"step{step}_{name}.bin", dtype=np.int64
                    ).reshape(shape)
                    expected = reference[f"step{step}_{name}"]
                    metric = metrics(raw, expected)
                else:
                    raise RuntimeError(f"unexpected ABI dtype: {spec['abi_dtype']}")
                saved[f"step{step}_{name}"] = raw
                item["outputs"][name] = metric

            current_key = saved[f"step{step}_updated_key"]
            current_value = saved[f"step{step}_updated_value"]
            flat_key = current_key.reshape(256, 4, 128)
            flat_value = current_value.reshape(256, 4, 128)
            previous_flat_key = previous_key.reshape(256, 4, 128)
            previous_flat_value = previous_value.reshape(256, 4, 128)
            addressed = 128 + step - 1
            unaddressed = np.ones(256, dtype=bool)
            unaddressed[addressed] = False
            item["unaddressed_key_exact"] = bool(
                np.array_equal(flat_key[unaddressed], previous_flat_key[unaddressed])
            )
            item["unaddressed_value_exact"] = bool(
                np.array_equal(flat_value[unaddressed], previous_flat_value[unaddressed])
            )
            item["all_outputs_finite"] = finite
            item["first_exact_mismatch"] = next(
                (spec["name"] for spec in specs if not item["outputs"][spec["name"]]["all_exact"]),
                None,
            )
            item["first_tolerance_failure"] = next(
                (
                    spec["name"]
                    for spec in specs
                    if not item["outputs"][spec["name"]]["all_within_tolerance"]
                ),
                None,
            )
            item["all_within_tolerance"] = item["first_tolerance_failure"] is None
            item["diagnostic_valid"] = (
                finite
                and item["unaddressed_key_exact"]
                and item["unaddressed_value_exact"]
                and int(saved[f"step{step}_next_position"][0]) == step
            )
            steps.append(item)
            previous_key, previous_value = current_key, current_value
    np.savez(args.output_npz, **saved)
    result = {
        "gate": "G4a Attempt 55d layer-0 BF16 materialization localization",
        "diagnostic_valid": all(item["diagnostic_valid"] for item in steps),
        "all_outputs_within_tolerance": all(item["all_within_tolerance"] for item in steps),
        "rtol": RTOL,
        "atol": ATOL,
        "reference_sha256": sha256(args.reference),
        "native_output_sha256": sha256(args.output_npz),
        "steps": steps,
        "claim_boundary": "Layer-0 localization only; this result cannot pass complete G4a.",
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "diagnostic_valid": result["diagnostic_valid"],
                "all_outputs_within_tolerance": result["all_outputs_within_tolerance"],
                "first_exact_mismatch": [item["first_exact_mismatch"] for item in steps],
                "first_tolerance_failure": [item["first_tolerance_failure"] for item in steps],
            },
            indent=2,
        )
    )
    if not result["diagnostic_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
