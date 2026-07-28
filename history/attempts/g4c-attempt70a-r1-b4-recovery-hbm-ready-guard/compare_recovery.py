#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BATCH = 4
VOCAB = 152064
MAX_STEPS = 8
CAPACITY = 8
BLOCK_SIZE = 128
EOS = 151645

CASES = {
    "invalid-max-steps": {"max_steps": 9, "sampling": 0, "graph": 0, "status": 201},
    "capacity-exceeded": {"max_steps": 2, "sampling": 0, "graph": 0, "status": 202},
    "unsupported-sampling": {"max_steps": 1, "sampling": 1, "graph": 0, "status": 205},
    "unsupported-graph": {"max_steps": 1, "sampling": 0, "graph": 1, "status": 206},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_runtime(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != "metric\tvalue":
        raise RuntimeError(f"invalid runtime file: {path}")
    return {key: int(value) for key, value in (line.split("\t") for line in lines[1:])}


def load_base(input_dir: Path, capacity_case: bool) -> dict[str, np.ndarray]:
    state = {
        "token": np.fromfile(input_dir / "token_id.bin", dtype=np.int64),
        "position": np.fromfile(input_dir / "position.bin", dtype=np.int64),
        "length": np.fromfile(input_dir / "sequence_length.bin", dtype=np.int32),
        "key": np.fromfile(input_dir / "key_cache.bin", dtype=np.uint16),
        "slot": np.fromfile(input_dir / "slot_mapping.bin", dtype=np.int32),
        "active": np.fromfile(input_dir / "active_mask.bin", dtype=np.int32),
        "block_table": np.fromfile(input_dir / "block_table.bin", dtype=np.int32),
        "value": np.fromfile(input_dir / "value_cache.bin", dtype=np.uint16),
    }
    if capacity_case:
        state["position"][0] = CAPACITY - 1
        state["length"][0] = CAPACITY
        state["slot"][0] = state["block_table"][0] * BLOCK_SIZE + CAPACITY - 1
    return state


def compare_case(name: str, spec: dict[str, int], input_dir: Path, output_dir: Path) -> dict:
    base = load_base(input_dir, name == "capacity-exceeded")
    outputs = {
        "token": np.fromfile(output_dir / "final_token.bin", dtype=np.int64),
        "position": np.fromfile(output_dir / "final_position.bin", dtype=np.int64),
        "length": np.fromfile(output_dir / "final_sequence_length.bin", dtype=np.int32),
        "key": np.fromfile(output_dir / "key_cache.bin", dtype=np.uint16),
        "slot": np.fromfile(output_dir / "final_slot_mapping.bin", dtype=np.int32),
        "active": np.fromfile(output_dir / "final_active_mask.bin", dtype=np.int32),
        "value": np.fromfile(output_dir / "value_cache.bin", dtype=np.uint16),
    }
    logits = np.fromfile(output_dir / "logits_history.bin", dtype=np.float32)
    tokens = np.fromfile(output_dir / "token_history.bin", dtype=np.int64)
    control = np.fromfile(output_dir / "control.bin", dtype=np.int32)
    runtime = read_runtime(output_dir / "runtime.tsv")

    expected_control = np.zeros(24, dtype=np.int32)
    expected_control[0] = spec["max_steps"]
    expected_control[1] = spec["sampling"]
    expected_control[2] = spec["graph"]
    expected_control[3] = spec["status"]
    expected_control[4] = 0
    expected_control[5] = 1
    expected_control[6:10] = EOS
    expected_control[10:14] = base["active"]
    expected_control[18:22] = 3
    active_count = int(np.count_nonzero(base["active"]))
    expected_control[22] = active_count
    expected_control[23] = active_count

    state_equal = {
        field: bool(np.array_equal(outputs[field], base[field]))
        for field in ("token", "position", "length", "key", "slot", "active", "value")
    }
    result = {
        "case": name,
        "expected_status": spec["status"],
        "observed_status": int(control[3]) if control.size == 24 else None,
        "control_exact": bool(np.array_equal(control, expected_control)),
        "model_calls_zero": bool(control.size == 24 and control[4] == 0),
        "fallback_flag_set": bool(control.size == 24 and control[5] == 1),
        "logits_zero": bool(logits.size == MAX_STEPS * BATCH * VOCAB and np.all(logits == 0)),
        "token_history_minus_one": bool(tokens.size == MAX_STEPS * BATCH and np.all(tokens == -1)),
        "state_exact": state_equal,
        "runtime": runtime,
        "one_feed_fetch": runtime["host_model_submissions"] == 1
        and runtime["feed_calls"] == 1
        and runtime["fetch_calls"] == 1,
        "control_sha256": sha256(output_dir / "control.bin"),
    }
    result["pass"] = bool(
        result["control_exact"]
        and result["model_calls_zero"]
        and result["fallback_flag_set"]
        and result["logits_zero"]
        and result["token_history_minus_one"]
        and all(state_equal.values())
        and result["one_feed_fetch"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.input_dir / "k8-all-active"
    cases = [
        compare_case(name, spec, source, args.output_dir / name)
        for name, spec in CASES.items()
    ]
    result = {
        "gate": "G4c Attempt 70a B=4 recovery",
        "pass": all(case["pass"] for case in cases),
        "cases": cases,
        "claim_boundary": (
            "This proves safe input-preserving fallback for four pre-execution "
            "control/metadata failures. Runtime model faults remain outside this attempt."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
