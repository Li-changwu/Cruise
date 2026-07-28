#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

from run_g2g_attempt1 import (
    ATOL,
    HEAD_DIM,
    REPEAT_COUNT,
    RTOL,
    custom_bmm,
    native_bmm,
    prepare_operands,
    sha256,
    to_numpy,
)
from vllm_ascend.utils import enable_custom_op


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"comparison shape mismatch actual={actual.shape} expected={expected.shape}"
        )
    actual_f32 = actual.astype(np.float32)
    expected_f32 = expected.astype(np.float32)
    difference = np.abs(actual_f32 - expected_f32)
    exact = np.equal(actual, expected)
    close = np.isclose(actual_f32, expected_f32, rtol=RTOL, atol=ATOL)
    return {
        "shape": list(actual.shape),
        "all_exact": bool(np.all(exact)),
        "all_within_tolerance": bool(np.all(close)),
        "max_abs_error": float(np.max(difference)),
        "exact_mismatch_count": int(exact.size - np.count_nonzero(exact)),
        "tolerance_failure_count": int(close.size - np.count_nonzero(close)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt8-output", type=Path, required=True)
    parser.add_argument("--eager-reference", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    enable_custom_op()
    registered = hasattr(torch.ops._C_ascend, "batch_matmul_transpose")
    if not registered:
        raise RuntimeError("batch_matmul_transpose is not registered")
    torch.npu.set_device(0)

    attempt8 = np.load(args.attempt8_output)
    eager = np.load(args.eager_reference)
    saved = {}
    steps = []
    for step in range(1, 5):
        tensor_a, tensor_b = prepare_operands(
            attempt8[f"step{step}_q_rope_bf16"],
            attempt8[f"step{step}_updated_key_bf16"],
        )
        native_raw = native_bmm(tensor_a, tensor_b)
        native_scaled = (native_raw / math.sqrt(HEAD_DIM)).unsqueeze(2)
        custom_repeats = [custom_bmm(tensor_a, tensor_b) for _ in range(REPEAT_COUNT)]
        torch.npu.synchronize()

        native_raw_np = to_numpy(native_raw)
        native_scaled_np = to_numpy(native_scaled)
        custom_raw_np = [to_numpy(value) for value in custom_repeats]
        custom_scaled_np = [
            to_numpy((value / math.sqrt(HEAD_DIM)).unsqueeze(2))
            for value in custom_repeats
        ]
        eager_qk = eager[f"step{step}_qk_scores"].astype(np.float32)
        deterministic = all(
            np.array_equal(custom_raw_np[0], value) for value in custom_raw_np[1:]
        )
        raw_vs_native = metrics(custom_raw_np[0], native_raw_np)
        native_vs_eager = metrics(native_scaled_np, eager_qk)
        custom_vs_eager = metrics(custom_scaled_np[0], eager_qk)
        step_pass = bool(
            deterministic
            and raw_vs_native["all_exact"]
            and native_vs_eager["all_exact"]
            and custom_vs_eager["all_exact"]
            and custom_vs_eager["all_within_tolerance"]
        )
        steps.append(
            {
                "step": step,
                "position": step - 1,
                "three_custom_runs_elementwise_deterministic": deterministic,
                "custom_raw_bmm_vs_native": raw_vs_native,
                "native_scaled_vs_eager": native_vs_eager,
                "custom_scaled_vs_eager": custom_vs_eager,
                "pass": step_pass,
            }
        )
        saved[f"step{step}_native_raw"] = native_raw_np
        saved[f"step{step}_native_scaled"] = native_scaled_np
        saved[f"step{step}_custom_raw"] = custom_raw_np[0]
        saved[f"step{step}_custom_scaled"] = custom_scaled_np[0]

    np.savez(args.output_npz, **saved)
    all_pass = bool(all(step["pass"] for step in steps))
    result = {
        "gate": "G2g attempt 44 physical-NPU-7 direct-launch control",
        "execution_success": True,
        "registered": registered,
        "physical_npu": 7,
        "rtol": RTOL,
        "atol": ATOL,
        "repeat_count": REPEAT_COUNT,
        "attempt8_output_sha256": sha256(args.attempt8_output),
        "eager_reference_sha256": sha256(args.eager_reference),
        "all_positions_pass": all_pass,
        "steps": steps,
        "claim_boundary": (
            "This is a Host-launched custom-kernel numerical screen. It does "
            "not prove GE/AIR embedding, Device-UDF recurrence, removal of "
            "per-token Host interaction, or positive latency."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("G2G_ATTEMPT3 " + json.dumps(result, ensure_ascii=True), flush=True)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
