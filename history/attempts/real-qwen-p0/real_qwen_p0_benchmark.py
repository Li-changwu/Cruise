#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import dataflow as df
import numpy as np


OUTPUT_NAMES = ("attention", "key_cache", "value_cache", "position")


def unpack_fetch(fetch_result):
    if not isinstance(fetch_result, tuple) or len(fetch_result) != 3:
        raise RuntimeError(f"unexpected fetch result: {type(fetch_result)}")
    outputs, _, ret_code = fetch_result
    if ret_code != 0 or len(outputs) != len(OUTPUT_NAMES):
        raise RuntimeError(
            f"fetch failed ret_code={ret_code} outputs={len(outputs)}"
        )
    return [np.asarray(output.numpy()).copy() for output in outputs]


def build_graph(args, route: str):
    flow_inputs = [df.FlowData() for _ in range(4)]
    graph_pp = df.GraphProcessPoint(
        df.Framework.MINDSPORE,
        str(args.air_path),
        compile_config_path=str(args.graph_config),
        name=f"real_qwen_p0_air_{route}_{args.n}",
    )
    node = df.FlowNode(
        input_num=4,
        output_num=4,
        name=f"real_qwen_p0_node_{route}_{args.n}",
    )
    if route == "host":
        node.add_process_point(graph_pp)
    else:
        func_pp = df.FuncProcessPoint(
            compile_config_path=str(args.func_config),
            name=f"real_qwen_p0_controller_{args.n}",
        )
        func_pp.set_init_param("loop_count", args.n)
        func_pp.add_invoked_closure("invoke_graph", graph_pp)
        node.add_process_point(func_pp)
    outputs = node(*flow_inputs)
    graph = df.FlowGraph(list(outputs), name=f"real_qwen_p0_{route}_{args.n}")
    return graph, flow_inputs


def feed_and_fetch(graph, flow_inputs, hidden, position, key, value):
    graph.feed_data(
        dict(zip(flow_inputs, [hidden, position, key, value], strict=True))
    )
    return unpack_fetch(graph.fetch_data(timeout=300000))


def run_host(graph, flow_inputs, initial):
    hidden, position, key, value = initial
    outputs = None
    for _ in range(initial.n):
        outputs = feed_and_fetch(
            graph, flow_inputs, hidden, position, key, value
        )
        _, key, value, position = outputs
    return outputs


def run_device(graph, flow_inputs, initial):
    hidden, position, key, value = initial
    return feed_and_fetch(graph, flow_inputs, hidden, position, key, value)


class InitialState:
    def __init__(self, reference, n):
        self.n = n
        self.hidden = reference["input_hidden_table"]
        self.position = reference["input_position"]
        self.key = reference["input_key_cache"]
        self.value = reference["input_value_cache"]

    def __iter__(self):
        return iter((self.hidden, self.position, self.key, self.value))


def measure(callable_):
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    outputs = callable_()
    cpu_us = (time.process_time_ns() - cpu_start) / 1_000
    wall_us = (time.perf_counter_ns() - wall_start) / 1_000
    return outputs, wall_us, cpu_us


def compare_outputs(host, device):
    exact = {
        name: bool(np.array_equal(host_value, device_value))
        for name, host_value, device_value in zip(
            OUTPUT_NAMES, host, device, strict=True
        )
    }
    max_abs = {
        name: float(
            np.max(
                np.abs(
                    host_value.astype(np.float32)
                    - device_value.astype(np.float32)
                )
            )
        )
        for name, host_value, device_value in zip(
            OUTPUT_NAMES, host, device, strict=True
        )
    }
    return exact, max_abs


def summary(samples):
    return {
        "wall_us_median": statistics.median(row["wall_us"] for row in samples),
        "host_cpu_us_median": statistics.median(
            row["host_cpu_us"] for row in samples
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--air-path", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--graph-config", type=Path, required=True)
    parser.add_argument("--func-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = np.load(args.reference)
    initial = InitialState(reference, args.n)
    options = {
        "ge.exec.deviceId": "0",
        "ge.exec.logicalDeviceClusterDeployMode": "SINGLE",
        "ge.exec.logicalDeviceId": "[0:0]",
    }
    df.init(options)
    try:
        host_graph, host_inputs = build_graph(args, "host")
        device_graph, device_inputs = build_graph(args, "device")
        host_call = lambda: run_host(host_graph, host_inputs, initial)
        device_call = lambda: run_device(device_graph, device_inputs, initial)

        for index in range(args.warmup):
            if index % 2 == 0:
                host_call()
                device_call()
            else:
                device_call()
                host_call()

        host_samples = []
        device_samples = []
        comparisons = []
        for rep in range(args.repetitions):
            if rep % 2 == 0:
                host_output, host_wall, host_cpu = measure(host_call)
                device_output, device_wall, device_cpu = measure(device_call)
            else:
                device_output, device_wall, device_cpu = measure(device_call)
                host_output, host_wall, host_cpu = measure(host_call)
            host_samples.append(
                {"rep": rep, "wall_us": host_wall, "host_cpu_us": host_cpu}
            )
            device_samples.append(
                {"rep": rep, "wall_us": device_wall, "host_cpu_us": device_cpu}
            )
            exact, max_abs = compare_outputs(host_output, device_output)
            comparisons.append(
                {
                    "rep": rep,
                    "all_exact": bool(all(exact.values())),
                    "tensor_exact": exact,
                    "max_abs": max_abs,
                    "host_final_position": int(host_output[3].reshape(-1)[0]),
                    "device_final_position": int(device_output[3].reshape(-1)[0]),
                }
            )
    finally:
        df.finalize()

    host_summary = summary(host_samples)
    device_summary = summary(device_samples)
    correct = bool(
        all(row["all_exact"] for row in comparisons)
        and all(row["host_final_position"] == args.n for row in comparisons)
        and all(row["device_final_position"] == args.n for row in comparisons)
    )
    result = {
        "schema_version": 1,
        "n": args.n,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "correct_host_vs_device": correct,
        "host": {
            "feed_calls_per_rep": args.n,
            "fetch_calls_per_rep": args.n,
            "samples": host_samples,
            "summary": host_summary,
        },
        "device": {
            "feed_calls_per_rep": 1,
            "fetch_calls_per_rep": 1,
            "samples": device_samples,
            "summary": device_summary,
        },
        "speedup": host_summary["wall_us_median"]
        / device_summary["wall_us_median"],
        "host_cpu_reduction_ratio": 1
        - device_summary["host_cpu_us_median"]
        / host_summary["host_cpu_us_median"],
        "comparisons": comparisons,
        "claim_boundary": (
            "The benchmark compares Host and Device-UDF recurrence of the same "
            "real-Qwen layer-0 attention/KV AIR. That AIR is not eager-equivalent "
            "under the frozen G2d tolerance and is not a full decoder or vLLM."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print("REAL_QWEN_P0 " + json.dumps(result, ensure_ascii=True), flush=True)
    if not correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
