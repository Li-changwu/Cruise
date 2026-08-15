#!/usr/bin/env python3
"""Validate fixed-K AIR recurrence against the one-step decoder AIR on NPU."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

from vllm_ascend_resident_epoch.contract import (
    CONTRACT_VERSION,
    ResidentEpochPlan,
    ResidentEpochRequest,
)
from vllm_ascend_resident_epoch.sidecar_backend import SidecarDataFlowEngine


BATCH_SIZE = 4
CAPACITY = 8
DEFAULT_EOS = 151645
INITIAL_TOKEN = 9707


def make_plan(
    *,
    label: str,
    generation_base: int,
    token_ids: list[int],
    position: int,
    max_steps: int,
    eos_token_ids: list[int],
    graph_variant: int,
    active_rows: tuple[int, ...] = tuple(range(BATCH_SIZE)),
) -> ResidentEpochPlan:
    if len(token_ids) != BATCH_SIZE or len(eos_token_ids) != BATCH_SIZE:
        raise ValueError("the static graph requires four token and EOS values")
    if not active_rows or any(row < 0 or row >= BATCH_SIZE for row in active_rows):
        raise ValueError("active rows must select at least one static-graph row")
    if len(set(active_rows)) != len(active_rows):
        raise ValueError("active rows must not contain duplicates")
    blocks_per_request = 1 if graph_variant == 0x10 else 2
    requests = tuple(
        ResidentEpochRequest(
            req_id=f"{label}-{row}",
            row=row,
            generation=generation_base + row,
            token_id=token_ids[row],
            position=position,
            sequence_length=position + 1,
            eos_token_id=eos_token_ids[row],
            scheduler_block_ids=tuple(
                row * blocks_per_request + local_block
                for local_block in range(blocks_per_request)
            ),
            device_block_ids=tuple(
                row * blocks_per_request + local_block
                for local_block in range(blocks_per_request)
            ),
        )
        for row in active_rows
    )
    plan = ResidentEpochPlan(
        version=CONTRACT_VERSION,
        graph_batch_size=BATCH_SIZE,
        max_steps=max_steps,
        logical_capacity=CAPACITY,
        requests=requests,
        active_mask=tuple(int(row in active_rows) for row in range(BATCH_SIZE)),
        graph_variant=graph_variant,
    )
    plan.validate()
    return plan


def run(engine: SidecarDataFlowEngine, plan: ResidentEpochPlan) -> dict[str, list[int]]:
    output = engine.execute(plan)
    if output.status != 0:
        raise RuntimeError(f"{plan.requests[0].req_id}: device status {output.status}")
    if output.model_calls != 1 or output.feed_calls != 1 or output.fetch_calls != 1:
        raise RuntimeError(
            f"{plan.requests[0].req_id}: expected one model/feed/fetch call, got "
            f"{output.model_calls}/{output.feed_calls}/{output.fetch_calls}"
        )
    if output.row_generations != plan.row_generations:
        raise RuntimeError("sidecar generation acknowledgement disagrees with plan")
    return output.token_ids


def assert_all_lengths(tokens: dict[str, list[int]], expected: int) -> None:
    actual = {request_id: len(value) for request_id, value in tokens.items()}
    if set(actual.values()) != {expected}:
        raise RuntimeError(f"unexpected token counts: {actual}")


@contextmanager
def sidecar_environment(
    *, air: Path, external_weights: Path, graph_config: Path, fixed_k: bool
):
    values = {
        "VLLM_ASCEND_RESIDENT_EPOCH_AIR": str(air.resolve(strict=True)),
        "VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS": str(
            external_weights.resolve(strict=True)
        ),
        "VLLM_ASCEND_RESIDENT_EPOCH_GRAPH_CONFIG": str(
            graph_config.resolve(strict=True)
        ),
        "VLLM_ASCEND_RESIDENT_EPOCH_K6": "1" if fixed_k else "0",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def run_one_step_reference() -> tuple[list[list[int]], list[int]]:
    engine = SidecarDataFlowEngine()
    try:
        current = [INITIAL_TOKEN] * BATCH_SIZE
        history = [[] for _ in range(BATCH_SIZE)]
        for position in range(6):
            plan = make_plan(
                label=f"single-{position}",
                generation_base=1,
                token_ids=current,
                position=position,
                max_steps=1,
                eos_token_ids=[DEFAULT_EOS] * BATCH_SIZE,
                graph_variant=0,
            )
            tokens = run(engine, plan)
            for row, request in enumerate(plan.requests):
                token = tokens[request.req_id]
                if len(token) != 1:
                    raise RuntimeError("one-step AIR emitted a non-unit token result")
                history[row].append(token[0])
                current[row] = token[0]
        continuation = run(
            engine,
            make_plan(
                label="single-continuation",
                generation_base=1,
                token_ids=current,
                position=6,
                max_steps=1,
                eos_token_ids=[DEFAULT_EOS] * BATCH_SIZE,
                graph_variant=0,
            ),
        )
        return history, [continuation[f"single-continuation-{row}"][0] for row in range(4)]
    finally:
        engine.close(force=True)


def run_fixed_k(
    reference_history: list[list[int]], reference_continuation: list[int]
) -> dict[str, object]:
    engine = SidecarDataFlowEngine()
    try:
        epoch = make_plan(
            label="k6",
            generation_base=101,
            token_ids=[INITIAL_TOKEN] * BATCH_SIZE,
            position=0,
            max_steps=6,
            eos_token_ids=[DEFAULT_EOS] * BATCH_SIZE,
            graph_variant=0x10,
        )
        epoch_tokens = run(engine, epoch)
        assert_all_lengths(epoch_tokens, 6)
        history = [epoch_tokens[f"k6-{row}"] for row in range(BATCH_SIZE)]
        if history != reference_history:
            raise RuntimeError(
                f"fixed-K token recurrence differs from six one-step calls: {history}"
            )

        continuation = run(
            engine,
            make_plan(
                label="k6-continuation",
                generation_base=101,
                token_ids=[history[row][-1] for row in range(BATCH_SIZE)],
                position=6,
                max_steps=1,
                eos_token_ids=[DEFAULT_EOS] * BATCH_SIZE,
                graph_variant=0x10,
            ),
        )
        continuation_tokens = [
            continuation[f"k6-continuation-{row}"][0] for row in range(BATCH_SIZE)
        ]
        if continuation_tokens != reference_continuation:
            raise RuntimeError(
                "fixed-K final KV/position state differs from one-step recurrence: "
                f"{continuation_tokens} != {reference_continuation}"
            )

        eos_token_ids = [history[0][0], DEFAULT_EOS, history[2][0], DEFAULT_EOS]
        eos_tokens = run(
            engine,
            make_plan(
                label="k6-eos",
                generation_base=201,
                token_ids=[INITIAL_TOKEN] * BATCH_SIZE,
                position=0,
                max_steps=6,
                eos_token_ids=eos_token_ids,
                graph_variant=0x10,
            ),
        )
        expected_counts = [1, 6, 1, 6]
        actual_counts = [len(eos_tokens[f"k6-eos-{row}"]) for row in range(4)]
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"per-row EOS did not stop Device recurrence: {actual_counts}"
            )
        if eos_tokens["k6-eos-0"] != history[0][:1] or eos_tokens[
            "k6-eos-2"
        ] != history[2][:1]:
            raise RuntimeError("EOS row emitted a token after its configured stop token")

        # The scheduler never continues after EOS. This direct recurrence check
        # proves the terminal row nevertheless retained exactly one KV update.
        eos_continuation = run(
            engine,
            make_plan(
                label="k6-eos-continuation",
                generation_base=201,
                token_ids=[history[row][0] for row in range(BATCH_SIZE)],
                position=1,
                max_steps=1,
                eos_token_ids=[DEFAULT_EOS] * BATCH_SIZE,
                graph_variant=0x10,
                # Rows 1 and 3 retained their six-step state from k6-eos;
                # only the EOS rows have a one-step resident state to resume.
                active_rows=(0, 2),
            ),
        )
        actual = [
            eos_continuation[f"k6-eos-continuation-{row}"][0] for row in (0, 2)
        ]
        expected = [history[row][1] for row in (0, 2)]
        if actual != expected:
            raise RuntimeError(f"EOS terminal KV state differs: {actual} != {expected}")
        return {
            "k6_history": history,
            "k6_continuation": continuation_tokens,
            "eos_token_counts": actual_counts,
            "eos_continuation": actual,
        }
    finally:
        engine.close(force=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-air", type=Path, required=True)
    parser.add_argument("--reference-weights", type=Path, required=True)
    parser.add_argument("--reference-graph-config", type=Path, required=True)
    parser.add_argument("--fixed-air", type=Path, required=True)
    parser.add_argument("--fixed-weights", type=Path, required=True)
    parser.add_argument("--fixed-graph-config", type=Path, required=True)
    args = parser.parse_args()

    args.output = args.output.resolve()
    for name in (
        "reference_air",
        "reference_weights",
        "reference_graph_config",
        "fixed_air",
        "fixed_weights",
        "fixed_graph_config",
    ):
        setattr(args, name, getattr(args, name).resolve(strict=True))

    # CANN writes kernel_meta relative to the current directory. Keep its
    # generated artifacts inside the caller-provided smoke output directory.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output.parent.resolve())

    with sidecar_environment(
        air=args.reference_air,
        external_weights=args.reference_weights,
        graph_config=args.reference_graph_config,
        fixed_k=False,
    ):
        one_step_history, one_step_continuation = run_one_step_reference()
    with sidecar_environment(
        air=args.fixed_air,
        external_weights=args.fixed_weights,
        graph_config=args.fixed_graph_config,
        fixed_k=True,
    ):
        fixed_k = run_fixed_k(one_step_history, one_step_continuation)
    result = {
        "gate": "M4b fixed-K Device recurrence and EOS state equivalence",
        "pass": True,
        "one_step_history": one_step_history,
        "one_step_continuation": one_step_continuation,
        **fixed_k,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
