#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3
VOCAB_SIZE = 152064
MAX_STEPS = 8
CACHE_SHAPE = (28, 2, 128, 4, 128)
FLAT_CACHE_SHAPE = (28, 256, 4, 128)
CONFIGURED_EOS = 151645


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


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    return metrics(bf16_to_float32(actual), bf16_to_float32(expected))


def read_runtime(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "metric\tvalue":
        raise RuntimeError(f"invalid runtime metadata: {path}")
    result = {}
    for line in lines[1:]:
        key, value = line.split("\t")
        result[key] = int(value)
    return result


def load_case(path: Path) -> dict:
    return {
        "logits": np.fromfile(path / "logits_history.bin", dtype=np.float32).reshape(
            MAX_STEPS, VOCAB_SIZE
        ),
        "tokens": np.fromfile(path / "token_history.bin", dtype=np.int64).reshape(
            MAX_STEPS
        ),
        "key": np.fromfile(path / "key_cache.bin", dtype=np.uint16).reshape(
            CACHE_SHAPE
        ),
        "value": np.fromfile(path / "value_cache.bin", dtype=np.uint16).reshape(
            CACHE_SHAPE
        ),
        "position": np.fromfile(
            path / "final_position.bin", dtype=np.int64
        ).reshape(1),
        "control": np.fromfile(path / "control.bin", dtype=np.int32).reshape(12),
        "runtime": read_runtime(path / "runtime.tsv"),
    }


def expected_control(max_steps: int, eos: int, tokens: np.ndarray) -> np.ndarray:
    visible = tokens[:max_steps]
    eos_positions = np.flatnonzero(visible == eos)
    if eos_positions.size:
        executed = int(eos_positions[0]) + 1
        finish_reason = 1
    else:
        executed = max_steps
        finish_reason = 2
    return np.asarray(
        [
            max_steps,
            eos,
            0,
            0,
            executed,
            finish_reason,
            0,
            int(tokens[executed - 1]),
            executed,
            executed + 1,
            executed,
            0,
        ],
        dtype=np.int32,
    )


def unaddressed_exact(
    final: np.ndarray,
    initial: np.ndarray,
    block_table: np.ndarray,
    executed: int,
) -> bool:
    addressed = []
    for position in range(executed):
        logical_block = position // 128
        offset = position % 128
        addressed.append(int(block_table[logical_block]) * 128 + offset)
    mask = np.ones(256, dtype=bool)
    mask[np.asarray(addressed, dtype=np.int64)] = False
    return bool(
        np.array_equal(
            final.reshape(FLAT_CACHE_SHAPE)[:, mask],
            initial.reshape(FLAT_CACHE_SHAPE)[:, mask],
        )
    )


def compare_case(
    name: str,
    host: dict,
    device: dict,
    max_steps: int,
    eos: int,
    initial_key: np.ndarray,
    initial_value: np.ndarray,
    block_table: np.ndarray,
) -> dict:
    expected = expected_control(max_steps, eos, host["tokens"])
    executed = int(expected[4])
    steps = []
    for step in range(executed):
        logits_metric = metrics(device["logits"][step], host["logits"][step])
        host_argmax = int(np.argmax(host["logits"][step]))
        device_argmax = int(np.argmax(device["logits"][step]))
        item = {
            "step": step + 1,
            "logits_host_device": logits_metric,
            "host_logits_finite": bool(np.all(np.isfinite(host["logits"][step]))),
            "device_logits_finite": bool(
                np.all(np.isfinite(device["logits"][step]))
            ),
            "host_token": int(host["tokens"][step]),
            "device_token": int(device["tokens"][step]),
            "host_argmax": host_argmax,
            "device_argmax": device_argmax,
        }
        item["pass"] = bool(
            logits_metric["all_within_tolerance"]
            and item["host_logits_finite"]
            and item["device_logits_finite"]
            and item["host_token"] == item["device_token"]
            and item["host_token"] == host_argmax
            and item["device_token"] == device_argmax
        )
        steps.append(item)

    key_metric = bf16_metrics(device["key"], host["key"])
    value_metric = bf16_metrics(device["value"], host["value"])
    result = {
        "case": name,
        "max_steps": max_steps,
        "eos_token": eos,
        "executed_steps": executed,
        "steps": steps,
        "token_history_equal": bool(
            np.array_equal(device["tokens"], host["tokens"])
        ),
        "padded_tokens_are_minus_one": bool(
            np.all(host["tokens"][executed:] == -1)
            and np.all(device["tokens"][executed:] == -1)
        ),
        "padded_logits_are_zero": bool(
            np.all(host["logits"][executed:] == 0)
            and np.all(device["logits"][executed:] == 0)
        ),
        "key_cache_host_device": key_metric,
        "value_cache_host_device": value_metric,
        "host_unaddressed_key_exact": unaddressed_exact(
            host["key"], initial_key, block_table, executed
        ),
        "device_unaddressed_key_exact": unaddressed_exact(
            device["key"], initial_key, block_table, executed
        ),
        "host_unaddressed_value_exact": unaddressed_exact(
            host["value"], initial_value, block_table, executed
        ),
        "device_unaddressed_value_exact": unaddressed_exact(
            device["value"], initial_value, block_table, executed
        ),
        "position_equal": bool(
            np.array_equal(device["position"], host["position"])
        ),
        "expected_final_position": executed,
        "host_final_position": int(host["position"][0]),
        "device_final_position": int(device["position"][0]),
        "control_equal": bool(
            np.array_equal(device["control"], host["control"])
        ),
        "host_control_expected": bool(np.array_equal(host["control"], expected)),
        "device_control_expected": bool(
            np.array_equal(device["control"], expected)
        ),
        "host_runtime": host["runtime"],
        "device_runtime": device["runtime"],
        "device_one_feed_fetch": bool(
            device["runtime"]["host_model_submissions"] == 1
            and device["runtime"]["feed_calls"] == 1
            and device["runtime"]["fetch_calls"] == 1
        ),
        "host_submission_count_expected": bool(
            host["runtime"]["host_model_submissions"] == executed
        ),
    }
    result["pass"] = bool(
        all(item["pass"] for item in steps)
        and result["token_history_equal"]
        and result["padded_tokens_are_minus_one"]
        and result["padded_logits_are_zero"]
        and key_metric["all_within_tolerance"]
        and value_metric["all_within_tolerance"]
        and result["host_unaddressed_key_exact"]
        and result["device_unaddressed_key_exact"]
        and result["host_unaddressed_value_exact"]
        and result["device_unaddressed_value_exact"]
        and result["position_equal"]
        and result["host_final_position"] == executed
        and result["device_final_position"] == executed
        and result["control_equal"]
        and result["host_control_expected"]
        and result["device_control_expected"]
        and result["device_one_feed_fetch"]
        and result["host_submission_count_expected"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-dir", type=Path, required=True)
    parser.add_argument("--device-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--eager-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial_key = np.fromfile(
        args.input_dir / "key_cache.bin", dtype=np.uint16
    ).reshape(CACHE_SHAPE)
    initial_value = np.fromfile(
        args.input_dir / "value_cache.bin", dtype=np.uint16
    ).reshape(CACHE_SHAPE)
    block_table = np.fromfile(
        args.input_dir / "block_table.bin", dtype=np.int32
    ).reshape(-1)
    early_eos = int(
        (args.host_dir / "early_eos_token.txt").read_text(encoding="utf-8").strip()
    )

    case_specs = [
        ("k1", 1, CONFIGURED_EOS),
        ("k2", 2, CONFIGURED_EOS),
        ("k4", 4, CONFIGURED_EOS),
        ("k8", 8, CONFIGURED_EOS),
        ("early-eos", 8, early_eos),
    ]
    cases = []
    loaded = {}
    for name, max_steps, eos in case_specs:
        host = load_case(args.host_dir / name)
        device = load_case(args.device_dir / name)
        loaded[name] = (host, device)
        cases.append(
            compare_case(
                name,
                host,
                device,
                max_steps,
                eos,
                initial_key,
                initial_value,
                block_table,
            )
        )

    k1_host, k1_device = loaded["k1"]
    with np.load(args.eager_reference) as eager:
        eager_continuity = {
            "host_logits_vs_eager": metrics(
                k1_host["logits"][0].reshape(1, 1, VOCAB_SIZE),
                eager["step1_logits"],
            ),
            "device_logits_vs_eager": metrics(
                k1_device["logits"][0].reshape(1, 1, VOCAB_SIZE),
                eager["step1_logits"],
            ),
            "host_key_vs_eager": bf16_metrics(
                k1_host["key"], eager["step1_key_cache_bits"]
            ),
            "device_key_vs_eager": bf16_metrics(
                k1_device["key"], eager["step1_key_cache_bits"]
            ),
            "host_value_vs_eager": bf16_metrics(
                k1_host["value"], eager["step1_value_cache_bits"]
            ),
            "device_value_vs_eager": bf16_metrics(
                k1_device["value"], eager["step1_value_cache_bits"]
            ),
            "host_greedy_equal": int(k1_host["tokens"][0])
            == int(np.argmax(eager["step1_logits"].reshape(-1))),
            "device_greedy_equal": int(k1_device["tokens"][0])
            == int(np.argmax(eager["step1_logits"].reshape(-1))),
            "host_position_equal": bool(
                np.array_equal(k1_host["position"], eager["step1_next_position"])
            ),
            "device_position_equal": bool(
                np.array_equal(k1_device["position"], eager["step1_next_position"])
            ),
        }
    eager_continuity["pass"] = bool(
        eager_continuity["host_logits_vs_eager"]["all_within_tolerance"]
        and eager_continuity["device_logits_vs_eager"]["all_within_tolerance"]
        and eager_continuity["host_key_vs_eager"]["all_within_tolerance"]
        and eager_continuity["device_key_vs_eager"]["all_within_tolerance"]
        and eager_continuity["host_value_vs_eager"]["all_within_tolerance"]
        and eager_continuity["device_value_vs_eager"]["all_within_tolerance"]
        and eager_continuity["host_greedy_equal"]
        and eager_continuity["device_greedy_equal"]
        and eager_continuity["host_position_equal"]
        and eager_continuity["device_position_equal"]
    )

    early = next(item for item in cases if item["case"] == "early-eos")
    result = {
        "gate": "G4b Attempt 66b-r1 B=1 device-resident real generation epoch",
        "pass": False,
        "rtol": RTOL,
        "atol": ATOL,
        "configured_eos": CONFIGURED_EOS,
        "controlled_early_eos_token": early_eos,
        "eager_reference_sha256": sha256(args.eager_reference),
        "eager_k1_continuity": eager_continuity,
        "cases": cases,
        "controlled_early_eos_pass": bool(
            early["pass"]
            and early["executed_steps"] == 1
            and early["host_control_expected"]
            and early["device_control_expected"]
        ),
        "claim_boundary": (
            "Correctness and one Feed/Fetch are tested for B=1. Performance, "
            "recovery, G4c batching, and vLLM integration are not passed here."
        ),
    }
    result["pass"] = bool(
        eager_continuity["pass"]
        and all(item["pass"] for item in cases)
        and result["controlled_early_eos_pass"]
    )
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
