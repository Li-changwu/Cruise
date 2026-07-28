#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu


DEFAULT_N = (1, 2, 4, 8, 16, 32)


def expected_state(n: int) -> list[int]:
    return [n, n, 1, n]


def summarize(samples: list[dict[str, int]]) -> dict[str, float]:
    keys = ("wall_us", "submit_cpu_us", "total_cpu_us")
    result: dict[str, float] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        result[f"{key}_median"] = statistics.median(values)
        result[f"{key}_min"] = min(values)
        result[f"{key}_max"] = max(values)
    return result


def measure_route(
    route: str,
    n: int,
    reps: int,
    warmups: int,
    state: torch.Tensor,
    initial: torch.Tensor,
    delta: torch.Tensor,
    graph: torch.npu.NPUGraph | None,
) -> dict:
    def reset() -> None:
        state.copy_(initial)
        torch.npu.synchronize()

    def submit() -> None:
        if route == "torch_eager":
            for _ in range(n):
                state.add_(delta)
        elif route == "npu_graph":
            if graph is None:
                raise RuntimeError("NPUGraph was not captured")
            for _ in range(n):
                graph.replay()
        else:
            raise ValueError(route)

    for _ in range(warmups):
        reset()
        submit()
        torch.npu.synchronize()
    reset()
    submit()
    torch.npu.synchronize()
    observed = state.cpu().tolist()
    expected = expected_state(n)
    if observed != expected:
        raise RuntimeError(
            f"{route} N={n} correctness failure: {observed} != {expected}"
        )

    samples = []
    for rep in range(1, reps + 1):
        reset()
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        submit()
        submit_cpu_end = time.process_time_ns()
        torch.npu.synchronize()
        cpu_end = time.process_time_ns()
        wall_end = time.perf_counter_ns()
        observed = state.cpu().tolist()
        if observed != expected:
            raise RuntimeError(
                f"{route} N={n} rep={rep} correctness failure: "
                f"{observed} != {expected}"
            )
        samples.append(
            {
                "rep": rep,
                "wall_us": (wall_end - wall_start) // 1000,
                "submit_cpu_us": (submit_cpu_end - cpu_start) // 1000,
                "total_cpu_us": (cpu_end - cpu_start) // 1000,
            }
        )
    return {
        "route": route,
        "n": n,
        "correct": True,
        "expected_state": expected,
        "observed_state": observed,
        "host_submission_calls": n,
        "host_completion_calls": 1,
        "samples": samples,
        "summary": summarize(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_N))
    args = parser.parse_args()
    if args.reps < 1 or args.warmups < 0 or any(n < 1 for n in args.n):
        raise ValueError("invalid benchmark argument")

    torch.npu.set_device(args.device)
    initial = torch.tensor([0, 0, 1, 0], dtype=torch.int32, device="npu")
    delta = torch.tensor([1, 1, 0, 1], dtype=torch.int32, device="npu")
    state = initial.clone()
    for _ in range(10):
        state.add_(delta)
    torch.npu.synchronize()
    state.copy_(initial)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        state.add_(delta)
    torch.npu.synchronize()
    state.copy_(initial)
    torch.npu.synchronize()

    cases = []
    for route in ("torch_eager", "npu_graph"):
        for n in args.n:
            case = measure_route(
                route,
                n,
                args.reps,
                args.warmups,
                state,
                initial,
                delta,
                graph,
            )
            cases.append(case)
            print(
                "P0_TORCH_RESULT "
                + json.dumps(
                    {
                        "route": route,
                        "n": n,
                        "correct": case["correct"],
                        **case["summary"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    result = {
        "schema_version": 1,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "physical_device": int(
            __import__("os").environ.get("ASCEND_RT_VISIBLE_DEVICES", "-1")
        ),
        "process_device": args.device,
        "n_values": args.n,
        "reps": args.reps,
        "warmups": args.warmups,
        "all_correct": all(case["correct"] for case in cases),
        "cases": cases,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
