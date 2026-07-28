#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RTOL = 5e-3
ATOL = 5e-3
BATCH_SIZE = 2
VOCAB_SIZE = 152064
MAX_STEPS = 8
BLOCK_SIZE = 128
CACHE_SHAPE = (28, 4, 128, 4, 128)
FLAT_CACHE_SHAPE = (28, 512, 4, 128)
CONFIGURED_EOS = 151645
CASE_SPECS = (
    ("k1-heterogeneous", 1),
    ("k2-heterogeneous", 2),
    ("k4-heterogeneous", 4),
    ("k8-both-active", 8),
    ("active-empty", 4),
    ("finished-active", 4),
    ("independent-early-eos", 4),
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


def bf16_float(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    return metrics(bf16_float(actual), bf16_float(expected))


def read_runtime(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "metric\tvalue":
        raise RuntimeError(f"invalid runtime file: {path}")
    return {key: int(value) for key, value in (line.split("\t") for line in lines[1:])}


def load_initial(path: Path) -> dict:
    return {
        "token": np.fromfile(path / "token_id.bin", dtype=np.int64).reshape(2),
        "position": np.fromfile(path / "position.bin", dtype=np.int64).reshape(2),
        "length": np.fromfile(path / "sequence_length.bin", dtype=np.int32).reshape(2),
        "key": np.fromfile(path / "key_cache.bin", dtype=np.uint16).reshape(CACHE_SHAPE),
        "slot": np.fromfile(path / "slot_mapping.bin", dtype=np.int32).reshape(2),
        "active": np.fromfile(path / "active_mask.bin", dtype=np.int32).reshape(2),
        "blocks": np.fromfile(path / "block_table.bin", dtype=np.int32).reshape(2, 2),
        "value": np.fromfile(path / "value_cache.bin", dtype=np.uint16).reshape(CACHE_SHAPE),
    }


def load_output(path: Path) -> dict:
    return {
        "logits": np.fromfile(path / "logits_history.bin", dtype=np.float32).reshape(
            MAX_STEPS, BATCH_SIZE, VOCAB_SIZE
        ),
        "tokens": np.fromfile(path / "token_history.bin", dtype=np.int64).reshape(
            MAX_STEPS, BATCH_SIZE
        ),
        "key": np.fromfile(path / "key_cache.bin", dtype=np.uint16).reshape(CACHE_SHAPE),
        "value": np.fromfile(path / "value_cache.bin", dtype=np.uint16).reshape(CACHE_SHAPE),
        "token": np.fromfile(path / "final_token.bin", dtype=np.int64).reshape(2),
        "position": np.fromfile(path / "final_position.bin", dtype=np.int64).reshape(2),
        "length": np.fromfile(path / "final_sequence_length.bin", dtype=np.int32).reshape(2),
        "slot": np.fromfile(path / "final_slot_mapping.bin", dtype=np.int32).reshape(2),
        "active": np.fromfile(path / "final_active_mask.bin", dtype=np.int32).reshape(2),
        "control": np.fromfile(path / "control.bin", dtype=np.int32).reshape(16),
        "runtime": read_runtime(path / "runtime.tsv"),
    }


def slot_for(blocks: np.ndarray, request: int, position: int) -> int:
    physical = int(blocks[request, position // BLOCK_SIZE])
    return physical * BLOCK_SIZE + position % BLOCK_SIZE


def expected_state(initial: dict, host: dict, max_steps: int, eos: np.ndarray) -> dict:
    executed = np.count_nonzero(host["tokens"] != -1, axis=0).astype(np.int32)
    reason = np.zeros(2, dtype=np.int32)
    final_active = initial["active"].copy()
    final_token = initial["token"].copy()
    final_position = initial["position"].copy()
    final_length = initial["length"].copy()
    final_slot = initial["slot"].copy()
    for request in range(2):
        if initial["active"][request] == 0:
            reason[request] = 4 if initial["length"][request] == 0 else 5
            continue
        count = int(executed[request])
        generated = host["tokens"][:count, request]
        if count:
            final_token[request] = generated[-1]
        final_position[request] += count
        final_length[request] = final_position[request] + 1
        final_slot[request] = (
            -1
            if final_position[request] == MAX_STEPS
            else slot_for(initial["blocks"], request, int(final_position[request]))
        )
        if count and int(generated[-1]) == int(eos[request]):
            reason[request] = 1
            final_active[request] = 0
        else:
            reason[request] = 2
            final_active[request] = 1
    model_calls = int(max(executed))
    control = np.asarray(
        [
            max_steps,
            0,
            0,
            0,
            model_calls,
            0,
            int(eos[0]),
            int(eos[1]),
            int(initial["active"][0]),
            int(initial["active"][1]),
            int(executed[0]),
            int(executed[1]),
            int(reason[0]),
            int(reason[1]),
            int(np.count_nonzero(initial["active"])),
            int(np.count_nonzero(final_active)),
        ],
        dtype=np.int32,
    )
    return {
        "executed": executed,
        "reason": reason,
        "token": final_token,
        "position": final_position,
        "length": final_length,
        "slot": final_slot,
        "active": final_active,
        "control": control,
        "model_calls": model_calls,
    }


def unaddressed_exact(final: np.ndarray, initial: np.ndarray, state: dict, executed: np.ndarray) -> bool:
    addressed = []
    for request in range(2):
        for offset in range(int(executed[request])):
            position = int(state["position"][request]) + offset
            addressed.append(slot_for(state["blocks"], request, position))
    mask = np.ones(512, dtype=bool)
    mask[np.asarray(addressed, dtype=np.int64)] = False
    return bool(
        np.array_equal(
            final.reshape(FLAT_CACHE_SHAPE)[:, mask],
            initial.reshape(FLAT_CACHE_SHAPE)[:, mask],
        )
    )


def inactive_cache_exact(final: np.ndarray, initial: np.ndarray, state: dict) -> bool:
    for request in range(2):
        if state["active"][request] != 0:
            continue
        blocks = state["blocks"][request].astype(np.int64)
        if not np.array_equal(final[:, blocks], initial[:, blocks]):
            return False
    return True


def compare_case(name: str, max_steps: int, eos: np.ndarray, initial: dict, host: dict, device: dict) -> dict:
    expected = expected_state(initial, host, max_steps, eos)
    steps = []
    for request in range(2):
        count = int(expected["executed"][request])
        for step in range(count):
            logit_metric = metrics(device["logits"][step, request], host["logits"][step, request])
            host_argmax = int(np.argmax(host["logits"][step, request]))
            device_argmax = int(np.argmax(device["logits"][step, request]))
            item = {
                "request": request,
                "step": step + 1,
                "logits_host_device": logit_metric,
                "host_finite": bool(np.all(np.isfinite(host["logits"][step, request]))),
                "device_finite": bool(np.all(np.isfinite(device["logits"][step, request]))),
                "host_token": int(host["tokens"][step, request]),
                "device_token": int(device["tokens"][step, request]),
                "host_argmax": host_argmax,
                "device_argmax": device_argmax,
            }
            item["pass"] = bool(
                logit_metric["all_within_tolerance"]
                and item["host_finite"]
                and item["device_finite"]
                and item["host_token"] == item["device_token"] == host_argmax == device_argmax
            )
            steps.append(item)

    padded_tokens = True
    padded_logits = True
    for request in range(2):
        count = int(expected["executed"][request])
        padded_tokens &= bool(np.all(host["tokens"][count:, request] == -1))
        padded_tokens &= bool(np.all(device["tokens"][count:, request] == -1))
        padded_logits &= bool(np.all(host["logits"][count:, request] == 0))
        padded_logits &= bool(np.all(device["logits"][count:, request] == 0))

    key_metric = bf16_metrics(device["key"], host["key"])
    value_metric = bf16_metrics(device["value"], host["value"])
    final_fields = {}
    for field in ("token", "position", "length", "slot", "active", "control"):
        final_fields[f"{field}_host_device_equal"] = bool(
            np.array_equal(host[field], device[field])
        )
        final_fields[f"host_{field}_expected"] = bool(
            np.array_equal(host[field], expected[field])
        )
        final_fields[f"device_{field}_expected"] = bool(
            np.array_equal(device[field], expected[field])
        )

    inactive_state_frozen = True
    for request in range(2):
        if initial["active"][request] == 0:
            for field in ("token", "position", "length", "slot"):
                inactive_state_frozen &= bool(
                    host[field][request] == initial[field][request]
                    and device[field][request] == initial[field][request]
                )
    result = {
        "case": name,
        "max_steps": max_steps,
        "eos_tokens": eos.tolist(),
        "initial_active_mask": initial["active"].tolist(),
        "initial_position": initial["position"].tolist(),
        "executed_steps": expected["executed"].tolist(),
        "finish_reason": expected["reason"].tolist(),
        "model_calls": expected["model_calls"],
        "steps": steps,
        "token_history_equal": bool(np.array_equal(host["tokens"], device["tokens"])),
        "padded_tokens_are_minus_one": padded_tokens,
        "padded_logits_are_zero": padded_logits,
        "key_cache_host_device": key_metric,
        "value_cache_host_device": value_metric,
        "host_unaddressed_key_exact": unaddressed_exact(host["key"], initial["key"], initial, expected["executed"]),
        "device_unaddressed_key_exact": unaddressed_exact(device["key"], initial["key"], initial, expected["executed"]),
        "host_unaddressed_value_exact": unaddressed_exact(host["value"], initial["value"], initial, expected["executed"]),
        "device_unaddressed_value_exact": unaddressed_exact(device["value"], initial["value"], initial, expected["executed"]),
        "host_inactive_request_key_exact": inactive_cache_exact(host["key"], initial["key"], initial),
        "device_inactive_request_key_exact": inactive_cache_exact(device["key"], initial["key"], initial),
        "host_inactive_request_value_exact": inactive_cache_exact(host["value"], initial["value"], initial),
        "device_inactive_request_value_exact": inactive_cache_exact(device["value"], initial["value"], initial),
        "inactive_token_length_position_slot_frozen": inactive_state_frozen,
        "final_fields": final_fields,
        "host_runtime": host["runtime"],
        "device_runtime": device["runtime"],
        "host_submission_count_expected": host["runtime"]["host_model_submissions"] == expected["model_calls"],
        "device_one_feed_fetch": bool(
            device["runtime"]["host_model_submissions"] == 1
            and device["runtime"]["feed_calls"] == 1
            and device["runtime"]["fetch_calls"] == 1
        ),
    }
    result["pass"] = bool(
        all(item["pass"] for item in steps)
        and result["token_history_equal"]
        and padded_tokens
        and padded_logits
        and key_metric["all_within_tolerance"]
        and value_metric["all_within_tolerance"]
        and result["host_unaddressed_key_exact"]
        and result["device_unaddressed_key_exact"]
        and result["host_unaddressed_value_exact"]
        and result["device_unaddressed_value_exact"]
        and result["host_inactive_request_key_exact"]
        and result["device_inactive_request_key_exact"]
        and result["host_inactive_request_value_exact"]
        and result["device_inactive_request_value_exact"]
        and inactive_state_frozen
        and all(final_fields.values())
        and result["host_submission_count_expected"]
        and result["device_one_feed_fetch"]
    )
    return result


def eager_continuity(host: dict, device: dict, reference_path: Path) -> dict:
    with np.load(reference_path) as reference:
        expected_logits = reference["case0_b2_logits"].reshape(2, VOCAB_SIZE)
        expected_key = reference["case0_b2_key_cache_bits"]
        expected_value = reference["case0_b2_value_cache_bits"]
        expected_position = reference["case0_b2_next_position"]
    result = {
        "host_logits_vs_b2_eager": metrics(host["logits"][0], expected_logits),
        "device_logits_vs_b2_eager": metrics(device["logits"][0], expected_logits),
        "host_key_vs_b2_eager": bf16_metrics(host["key"], expected_key),
        "device_key_vs_b2_eager": bf16_metrics(device["key"], expected_key),
        "host_value_vs_b2_eager": bf16_metrics(host["value"], expected_value),
        "device_value_vs_b2_eager": bf16_metrics(device["value"], expected_value),
        "host_position_equal": bool(np.array_equal(host["position"], expected_position)),
        "device_position_equal": bool(np.array_equal(device["position"], expected_position)),
        "host_greedy_equal": bool(
            np.array_equal(host["tokens"][0], np.argmax(expected_logits, axis=1))
        ),
        "device_greedy_equal": bool(
            np.array_equal(device["tokens"][0], np.argmax(expected_logits, axis=1))
        ),
    }
    result["pass"] = bool(
        all(
            result[key]["all_within_tolerance"]
            for key in (
                "host_logits_vs_b2_eager",
                "device_logits_vs_b2_eager",
                "host_key_vs_b2_eager",
                "device_key_vs_b2_eager",
                "host_value_vs_b2_eager",
                "device_value_vs_b2_eager",
            )
        )
        and result["host_position_equal"]
        and result["device_position_equal"]
        and result["host_greedy_equal"]
        and result["device_greedy_equal"]
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
    early_values = np.asarray(
        [int(value) for value in (args.host_dir / "early_eos_tokens.txt").read_text(encoding="utf-8").split()],
        dtype=np.int32,
    )
    if early_values.shape != (2,):
        raise RuntimeError("expected two early EOS values")
    cases = []
    loaded = {}
    for name, max_steps in CASE_SPECS:
        initial = load_initial(args.input_dir / name)
        host = load_output(args.host_dir / name)
        device = load_output(args.device_dir / name)
        eos = early_values if name == "independent-early-eos" else np.asarray(
            [CONFIGURED_EOS, CONFIGURED_EOS], dtype=np.int32
        )
        cases.append(compare_case(name, max_steps, eos, initial, host, device))
        loaded[name] = (host, device)
    continuity = eager_continuity(
        loaded["k1-heterogeneous"][0],
        loaded["k1-heterogeneous"][1],
        args.eager_reference,
    )
    early = next(case for case in cases if case["case"] == "independent-early-eos")
    independent_eos_pass = bool(
        early["executed_steps"][0] == 1
        and early["finish_reason"][0] == 1
        and early["executed_steps"][1] == 4
        and early["finish_reason"][1] == 2
    )
    result = {
        "gate": "G4c Attempt 68a B=2 Device UDF resident generation epoch",
        "pass": all(case["pass"] for case in cases)
        and continuity["pass"]
        and independent_eos_pass,
        "rtol": RTOL,
        "atol": ATOL,
        "eager_reference_sha256": sha256(args.eager_reference),
        "early_eos_tokens": early_values.tolist(),
        "independent_eos_pass": independent_eos_pass,
        "eager_continuity": continuity,
        "cases": cases,
        "claim_boundary": (
            "This closes the B=2 correctness and one-Feed/one-Fetch resident "
            "epoch sub-gate only. B=4, stable performance benefit, recovery, "
            "and vLLM integration remain open."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
