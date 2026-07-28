#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from batched_decoder_step import (
    ATOL,
    BATCH_SIZE,
    BLOCK_SIZE,
    HEAD_DIM,
    LOGICAL_CAPACITY,
    NUM_KV_HEADS,
    NUM_LAYERS,
    PHYSICAL_BLOCKS,
    PHYSICAL_BLOCKS_PER_REQUEST,
    RTOL,
    TILING_WORDS,
    VOCAB_SIZE,
    PagedQwenDecoderStep,
    audit_checkpoint,
    bf16_bits,
    bits_to_bf16,
    load_checkpoint,
    register_custom_ops,
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


def make_cache(seed: int) -> tuple[np.ndarray, np.ndarray]:
    shape = (
        NUM_LAYERS,
        PHYSICAL_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    key = (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.125).to(
        torch.bfloat16
    )
    value = (
        torch.randn(shape, generator=generator, dtype=torch.float32) * 0.125
    ).to(torch.bfloat16)
    return bf16_bits(key), bf16_bits(value)


def block_table() -> np.ndarray:
    rows = []
    for request in range(BATCH_SIZE):
        base = request * PHYSICAL_BLOCKS_PER_REQUEST
        rows.append([base + 1, base])
    return np.asarray(rows, dtype=np.int32)


def slot_for(request: int, position: int) -> int:
    physical_block = int(block_table()[request, position // BLOCK_SIZE])
    return physical_block * BLOCK_SIZE + position % BLOCK_SIZE


def case_specs(token_ids: np.ndarray) -> list[dict]:
    return [
        {
            "name": "both-active-heterogeneous",
            "token": np.asarray([[token_ids[0]], [token_ids[2]]], dtype=np.int64),
            "position": np.asarray([0, 2], dtype=np.int64),
            "sequence_length": np.asarray([[1], [3]], dtype=np.int32),
            "active_mask": np.asarray([1, 1], dtype=np.int32),
        },
        {
            "name": "active-plus-empty",
            "token": np.asarray([[token_ids[1]], [0]], dtype=np.int64),
            "position": np.asarray([1, 0], dtype=np.int64),
            "sequence_length": np.asarray([[2], [0]], dtype=np.int32),
            "active_mask": np.asarray([1, 0], dtype=np.int32),
        },
        {
            "name": "finished-plus-active",
            "token": np.asarray([[token_ids[3]], [token_ids[1]]], dtype=np.int64),
            "position": np.asarray([3, 1], dtype=np.int64),
            "sequence_length": np.asarray([[4], [2]], dtype=np.int32),
            "active_mask": np.asarray([0, 1], dtype=np.int32),
        },
    ]


def add_case_state(specs: list[dict]) -> None:
    table = block_table()
    for index, case in enumerate(specs):
        key, value = make_cache(670100 + index)
        case["block_table"] = table.copy()
        case["slot_mapping"] = np.asarray(
            [slot_for(request, int(case["position"][request])) for request in range(BATCH_SIZE)],
            dtype=np.int32,
        )
        case["input_key_bits"] = key
        case["input_value_bits"] = value


def local_cache(global_bits: np.ndarray, request: int) -> np.ndarray:
    begin = request * PHYSICAL_BLOCKS_PER_REQUEST
    end = begin + PHYSICAL_BLOCKS_PER_REQUEST
    return np.ascontiguousarray(global_bits[:, begin:end])


def run_b1_oracles(
    model_dir: Path,
    specs: list[dict],
    tiling: torch.Tensor,
) -> list[dict]:
    model = PagedQwenDecoderStep(
        load_checkpoint(model_dir), batch_size=1, physical_blocks=2
    ).eval().npu()
    outputs = []
    with torch.no_grad():
        for case in specs:
            expected_key = case["input_key_bits"].copy()
            expected_value = case["input_value_bits"].copy()
            expected_logits = np.full((BATCH_SIZE, 1, VOCAB_SIZE), np.nan, dtype=np.float32)
            expected_greedy = np.full((BATCH_SIZE,), -1, dtype=np.int64)
            expected_position = case["position"].copy()
            request_records = []
            for request in range(BATCH_SIZE):
                if case["active_mask"][request] == 0:
                    request_records.append({"request": request, "active": False})
                    continue
                key = bits_to_bf16(local_cache(case["input_key_bits"], request)).npu()
                value = bits_to_bf16(local_cache(case["input_value_bits"], request)).npu()
                token = torch.from_numpy(case["token"][request : request + 1].copy()).npu()
                position = torch.from_numpy(
                    case["position"][request : request + 1].copy()
                ).npu()
                sequence_length = torch.from_numpy(
                    case["sequence_length"][request : request + 1].copy()
                ).npu()
                local_slot = case["slot_mapping"][request] - (
                    request * PHYSICAL_BLOCKS_PER_REQUEST * BLOCK_SIZE
                )
                slot_mapping = torch.tensor([local_slot], dtype=torch.int32, device="npu")
                local_table = torch.tensor([[1, 0]], dtype=torch.int32, device="npu")
                active = torch.ones((1,), dtype=torch.int32, device="npu")
                logits, next_key, next_value, next_position = model(
                    token,
                    position,
                    sequence_length,
                    local_table,
                    slot_mapping,
                    key,
                    value,
                    tiling,
                    active,
                )
                torch.npu.synchronize()
                logits_np = logits.detach().cpu().numpy().copy()
                next_key_bits = bf16_bits(next_key)
                next_value_bits = bf16_bits(next_value)
                begin = request * PHYSICAL_BLOCKS_PER_REQUEST
                end = begin + PHYSICAL_BLOCKS_PER_REQUEST
                expected_key[:, begin:end] = next_key_bits
                expected_value[:, begin:end] = next_value_bits
                expected_logits[request] = logits_np[0]
                expected_greedy[request] = int(np.argmax(logits_np.reshape(-1)))
                expected_position[request] = int(next_position.item())
                request_records.append(
                    {
                        "request": request,
                        "active": True,
                        "local_slot": int(local_slot),
                        "greedy": int(expected_greedy[request]),
                    }
                )
                del key, value, token, position, sequence_length
                del logits, next_key, next_value, next_position
            outputs.append(
                {
                    "logits": expected_logits,
                    "key_bits": expected_key,
                    "value_bits": expected_value,
                    "next_position": expected_position,
                    "greedy": expected_greedy,
                    "requests": request_records,
                }
            )
    del model
    gc.collect()
    torch.npu.empty_cache()
    return outputs


def mutation_mask(case: dict) -> np.ndarray:
    slots = PHYSICAL_BLOCKS * BLOCK_SIZE
    mask = np.zeros((slots,), dtype=bool)
    for request in range(BATCH_SIZE):
        if case["active_mask"][request] != 0:
            mask[int(case["slot_mapping"][request])] = True
    return mask


def run_b2(
    model_dir: Path,
    specs: list[dict],
    oracle: list[dict],
    tiling: torch.Tensor,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    model = PagedQwenDecoderStep(
        load_checkpoint(model_dir),
        batch_size=BATCH_SIZE,
        physical_blocks=PHYSICAL_BLOCKS,
    ).eval().npu()
    results = []
    archive = {}
    with torch.no_grad():
        for case_index, (case, expected) in enumerate(zip(specs, oracle)):
            prefix = f"case{case_index}"
            logits, key, value, next_position = model(
                torch.from_numpy(case["token"].copy()).npu(),
                torch.from_numpy(case["position"].copy()).npu(),
                torch.from_numpy(case["sequence_length"].copy()).npu(),
                torch.from_numpy(case["block_table"].copy()).npu(),
                torch.from_numpy(case["slot_mapping"].copy()).npu(),
                bits_to_bf16(case["input_key_bits"]).npu(),
                bits_to_bf16(case["input_value_bits"]).npu(),
                tiling,
                torch.from_numpy(case["active_mask"].copy()).npu(),
            )
            torch.npu.synchronize()
            logits_np = logits.detach().cpu().numpy().copy()
            key_bits = bf16_bits(key)
            value_bits = bf16_bits(value)
            position_np = next_position.detach().cpu().numpy().copy()
            active_logits = []
            greedy_equal = []
            logits_finite = []
            for request in range(BATCH_SIZE):
                if case["active_mask"][request] == 0:
                    continue
                logit_metric = metrics(logits_np[request], expected["logits"][request])
                active_logits.append({"request": request, **logit_metric})
                actual_greedy = int(np.argmax(logits_np[request].reshape(-1)))
                greedy_equal.append(actual_greedy == int(expected["greedy"][request]))
                logits_finite.append(bool(np.all(np.isfinite(logits_np[request]))))
            key_metric = bf16_metrics(key_bits, expected["key_bits"])
            value_metric = bf16_metrics(value_bits, expected["value_bits"])
            mutable = mutation_mask(case)
            flat_key = key_bits.reshape(
                NUM_LAYERS,
                PHYSICAL_BLOCKS * BLOCK_SIZE,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            flat_value = value_bits.reshape(flat_key.shape)
            input_flat_key = case["input_key_bits"].reshape(flat_key.shape)
            input_flat_value = case["input_value_bits"].reshape(flat_value.shape)
            unaddressed_key_exact = bool(
                np.array_equal(flat_key[:, ~mutable], input_flat_key[:, ~mutable])
            )
            unaddressed_value_exact = bool(
                np.array_equal(flat_value[:, ~mutable], input_flat_value[:, ~mutable])
            )
            inactive_request_cache_exact = []
            for request in range(BATCH_SIZE):
                if case["active_mask"][request] != 0:
                    continue
                begin = request * PHYSICAL_BLOCKS_PER_REQUEST
                end = begin + PHYSICAL_BLOCKS_PER_REQUEST
                inactive_request_cache_exact.append(
                    bool(
                        np.array_equal(
                            key_bits[:, begin:end], case["input_key_bits"][:, begin:end]
                        )
                        and np.array_equal(
                            value_bits[:, begin:end], case["input_value_bits"][:, begin:end]
                        )
                    )
                )
            item = {
                "case": case["name"],
                "token": case["token"].reshape(-1).tolist(),
                "position": case["position"].tolist(),
                "sequence_length": case["sequence_length"].reshape(-1).tolist(),
                "slot_mapping": case["slot_mapping"].tolist(),
                "active_mask": case["active_mask"].tolist(),
                "b1_requests": expected["requests"],
                "active_logits_vs_independent_b1": active_logits,
                "active_logits_finite": all(logits_finite),
                "active_greedy_equal": all(greedy_equal),
                "key_cache_vs_packed_b1": key_metric,
                "value_cache_vs_packed_b1": value_metric,
                "next_position_actual": position_np.tolist(),
                "next_position_expected": expected["next_position"].tolist(),
                "next_position_equal": bool(
                    np.array_equal(position_np, expected["next_position"])
                ),
                "unaddressed_key_elementwise_exact": unaddressed_key_exact,
                "unaddressed_value_elementwise_exact": unaddressed_value_exact,
                "inactive_request_cache_elementwise_exact": all(
                    inactive_request_cache_exact
                ),
            }
            item["pass"] = bool(
                all(metric["all_within_tolerance"] for metric in active_logits)
                and item["active_logits_finite"]
                and item["active_greedy_equal"]
                and key_metric["all_within_tolerance"]
                and value_metric["all_within_tolerance"]
                and item["next_position_equal"]
                and unaddressed_key_exact
                and unaddressed_value_exact
                and item["inactive_request_cache_elementwise_exact"]
            )
            results.append(item)
            archive[f"{prefix}_token"] = case["token"]
            archive[f"{prefix}_position"] = case["position"]
            archive[f"{prefix}_sequence_length"] = case["sequence_length"]
            archive[f"{prefix}_block_table"] = case["block_table"]
            archive[f"{prefix}_slot_mapping"] = case["slot_mapping"]
            archive[f"{prefix}_active_mask"] = case["active_mask"]
            archive[f"{prefix}_input_key_cache_bits"] = case["input_key_bits"]
            archive[f"{prefix}_input_value_cache_bits"] = case["input_value_bits"]
            archive[f"{prefix}_b2_logits"] = logits_np
            archive[f"{prefix}_b2_key_cache_bits"] = key_bits
            archive[f"{prefix}_b2_value_cache_bits"] = value_bits
            archive[f"{prefix}_b2_next_position"] = position_np
            archive[f"{prefix}_b1_logits"] = expected["logits"]
            archive[f"{prefix}_b1_key_cache_bits"] = expected["key_bits"]
            archive[f"{prefix}_b1_value_cache_bits"] = expected["value_bits"]
            archive[f"{prefix}_b1_next_position"] = expected["next_position"]
            del logits, key, value, next_position
    del model
    gc.collect()
    torch.npu.empty_cache()
    return results, archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--g4a-reference", type=Path, required=True)
    parser.add_argument("--exact-qk-source", type=Path, required=True)
    parser.add_argument("--barrier-source", type=Path, required=True)
    parser.add_argument("--materialize-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = audit_checkpoint(args.model_dir)
    (args.output_dir / "checkpoint-audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    if not audit["valid"]:
        raise SystemExit(2)
    register_custom_ops(
        args.exact_qk_source, args.barrier_source, args.materialize_source
    )
    torch.npu.set_device(0)
    with np.load(args.g4a_reference) as reference:
        token_ids = reference["token_ids"].copy()
    if token_ids.shape != (4,):
        raise RuntimeError(f"unexpected frozen token ids: {token_ids.shape}")
    specs = case_specs(token_ids)
    add_case_state(specs)
    tiling_cpu = torch.from_numpy(
        np.asarray(TILING_WORDS, dtype="<u4").view(np.uint8).copy()
    )
    tiling = tiling_cpu.npu()
    oracle = run_b1_oracles(args.model_dir, specs, tiling)
    cases, archive = run_b2(args.model_dir, specs, oracle, tiling)
    archive["token_ids"] = token_ids
    archive["tiling"] = tiling_cpu.numpy()
    reference_path = args.output_dir / "attempt67a-b2-eager-reference.npz"
    np.savez(reference_path, **archive)
    result = {
        "gate": "G4c Attempt 67a B=2 batched eager versus independent B=1 oracle",
        "pass": all(case["pass"] for case in cases),
        "model_revision": args.model_dir.name,
        "g4a_reference_sha256": sha256(args.g4a_reference),
        "reference_sha256": sha256(reference_path),
        "batch_size": BATCH_SIZE,
        "physical_blocks_per_request": PHYSICAL_BLOCKS_PER_REQUEST,
        "physical_blocks": PHYSICAL_BLOCKS,
        "block_table": block_table().tolist(),
        "logical_capacity": LOGICAL_CAPACITY,
        "rtol": RTOL,
        "atol": ATOL,
        "cases": cases,
        "abi_semantics": {
            "token": [BATCH_SIZE, 1],
            "position": [BATCH_SIZE],
            "sequence_length": [BATCH_SIZE, 1],
            "block_table": [BATCH_SIZE, PHYSICAL_BLOCKS_PER_REQUEST],
            "slot_mapping": [BATCH_SIZE],
            "key_cache": [
                NUM_LAYERS,
                PHYSICAL_BLOCKS,
                BLOCK_SIZE,
                NUM_KV_HEADS,
                HEAD_DIM,
            ],
            "value_cache": [
                NUM_LAYERS,
                PHYSICAL_BLOCKS,
                BLOCK_SIZE,
                NUM_KV_HEADS,
                HEAD_DIM,
            ],
            "explicit_tiling": [72],
            "active_mask": [BATCH_SIZE],
            "logits": [BATCH_SIZE, 1, VOCAB_SIZE],
            "next_position": [BATCH_SIZE],
        },
        "claim_boundary": (
            "This closes only B=2 eager batching semantics. AIR, Device UDF, "
            "per-request EOS, B=4, performance, recovery, and vLLM integration "
            "remain open."
        ),
    }
    result_path = args.output_dir / "attempt67a-b2-eager-result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("G4C_ATTEMPT67A " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
