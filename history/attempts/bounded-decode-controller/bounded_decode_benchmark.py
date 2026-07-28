#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import dataflow as df
import numpy as np


MODEL_OUTPUT_NAMES = ("attention", "key_cache", "value_cache", "position")
OUTPUT_NAMES = MODEL_OUTPUT_NAMES + ("control",)
VOCAB_SIZE = 151936
FINISH_EOS = 1
FINISH_MAX_STEPS = 2


@dataclass(frozen=True)
class Scenario:
    name: str
    max_steps: int
    eos_token: int
    eos_after_step: int
    graph_switch_step: int
    token_seed: int = 7
    token_stride: int = 13

    def input_control(self):
        return np.asarray(
            [
                self.max_steps,
                self.eos_token,
                self.eos_after_step,
                self.graph_switch_step,
                self.token_seed,
                self.token_stride,
            ],
            dtype=np.int32,
        )


SCENARIOS = (
    Scenario("max1", 1, 151643, 0, 1),
    Scenario("eos2_of4", 4, 151643, 2, 1),
    Scenario("eos3_of4", 4, 151643, 3, 2),
    Scenario("max4_switch2", 4, 151643, 0, 2),
)


def unpack_fetch(fetch_result, expected_outputs):
    if not isinstance(fetch_result, tuple) or len(fetch_result) != 3:
        raise RuntimeError(f"unexpected fetch result: {type(fetch_result)}")
    outputs, _, ret_code = fetch_result
    if ret_code != 0 or len(outputs) != expected_outputs:
        raise RuntimeError(
            f"fetch failed ret_code={ret_code} outputs={len(outputs)}"
        )
    return [np.asarray(output.numpy()).copy() for output in outputs]


def make_graph_pp(args, name):
    return df.GraphProcessPoint(
        df.Framework.MINDSPORE,
        str(args.air_path),
        compile_config_path=str(args.graph_config),
        name=name,
    )


def build_host_graph(args, variant):
    flow_inputs = [df.FlowData() for _ in range(4)]
    graph_pp = make_graph_pp(args, f"host_decode_graph_{variant}")
    node = df.FlowNode(
        input_num=4,
        output_num=4,
        name=f"host_decode_node_{variant}",
    )
    node.add_process_point(graph_pp)
    outputs = node(*flow_inputs)
    graph = df.FlowGraph(list(outputs), name=f"host_decode_{variant}")
    return graph, flow_inputs


def build_device_graph(args):
    flow_inputs = [df.FlowData() for _ in range(5)]
    graph0 = make_graph_pp(args, "device_decode_graph_0")
    graph1 = make_graph_pp(args, "device_decode_graph_1")
    controller = df.FuncProcessPoint(
        compile_config_path=str(args.func_config),
        name="bounded_decode_controller_pp",
    )
    controller.add_invoked_closure("decode_graph_0", graph0)
    controller.add_invoked_closure("decode_graph_1", graph1)
    node = df.FlowNode(
        input_num=5,
        output_num=5,
        name="bounded_decode_controller_node",
    )
    node.add_process_point(controller)
    outputs = node(*flow_inputs)
    graph = df.FlowGraph(list(outputs), name="bounded_decode_control_flow")
    return graph, flow_inputs


def feed_and_fetch(graph, flow_inputs, values, expected_outputs):
    graph.feed_data(dict(zip(flow_inputs, values, strict=True)))
    return unpack_fetch(graph.fetch_data(timeout=300000), expected_outputs)


def synthetic_token(scenario, executed_steps):
    token = (
        scenario.token_seed + scenario.token_stride * executed_steps
    ) % VOCAB_SIZE
    if scenario.eos_after_step > 0 and executed_steps == scenario.eos_after_step:
        token = scenario.eos_token
    return token


def make_final_control(scenario, executed, token, reason, graph0, graph1, position):
    return np.asarray(
        [
            *scenario.input_control().tolist(),
            executed,
            token,
            reason,
            graph0,
            graph1,
            position,
        ],
        dtype=np.int32,
    )


def run_host(host_graphs, host_inputs, initial, scenario):
    hidden, position, key, value = initial
    outputs = None
    executed = 0
    graph0_calls = 0
    graph1_calls = 0
    final_token = scenario.token_seed
    finish_reason = FINISH_MAX_STEPS
    while executed < scenario.max_steps:
        variant = 0 if executed < scenario.graph_switch_step else 1
        outputs = feed_and_fetch(
            host_graphs[variant],
            host_inputs[variant],
            (hidden, position, key, value),
            4,
        )
        _, key, value, position = outputs
        executed += 1
        if variant == 0:
            graph0_calls += 1
        else:
            graph1_calls += 1
        final_token = synthetic_token(scenario, executed)
        if final_token == scenario.eos_token:
            finish_reason = FINISH_EOS
            break
        if executed >= scenario.max_steps:
            finish_reason = FINISH_MAX_STEPS
            break
    final_position = int(np.asarray(outputs[3]).reshape(-1)[0])
    control = make_final_control(
        scenario,
        executed,
        final_token,
        finish_reason,
        graph0_calls,
        graph1_calls,
        final_position,
    )
    return outputs + [control]


def run_device(device_graph, device_inputs, initial, scenario):
    return feed_and_fetch(
        device_graph,
        device_inputs,
        (*initial, scenario.input_control()),
        5,
    )


def measure(callable_):
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    outputs = callable_()
    cpu_us = (time.process_time_ns() - cpu_start) / 1_000
    wall_us = (time.perf_counter_ns() - wall_start) / 1_000
    return outputs, wall_us, cpu_us


def summarize_samples(samples):
    return {
        "wall_us_median": statistics.median(row["wall_us"] for row in samples),
        "wall_us_iqr": statistics.quantiles(
            [row["wall_us"] for row in samples], n=4, method="inclusive"
        )[2]
        - statistics.quantiles(
            [row["wall_us"] for row in samples], n=4, method="inclusive"
        )[0],
        "host_cpu_us_median": statistics.median(
            row["host_cpu_us"] for row in samples
        ),
    }


def compare_outputs(host, device):
    tensor_exact = {
        name: bool(np.array_equal(host_value, device_value))
        for name, host_value, device_value in zip(
            OUTPUT_NAMES, host, device, strict=True
        )
    }
    max_abs = {
        name: float(
            np.max(
                np.abs(
                    host_value.astype(np.float64)
                    - device_value.astype(np.float64)
                )
            )
        )
        for name, host_value, device_value in zip(
            OUTPUT_NAMES, host, device, strict=True
        )
    }
    return tensor_exact, max_abs


def expected_execution(scenario):
    if scenario.eos_after_step > 0:
        return scenario.eos_after_step, FINISH_EOS
    return scenario.max_steps, FINISH_MAX_STEPS


def run_scenario(
    scenario,
    args,
    host_graphs,
    host_inputs,
    device_graph,
    device_inputs,
    initial,
):
    host_call = lambda: run_host(host_graphs, host_inputs, initial, scenario)
    device_call = lambda: run_device(
        device_graph, device_inputs, initial, scenario
    )
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
        tensor_exact, max_abs = compare_outputs(host_output, device_output)
        comparisons.append(
            {
                "rep": rep,
                "all_exact": bool(all(tensor_exact.values())),
                "tensor_exact": tensor_exact,
                "max_abs": max_abs,
                "host_control": host_output[4].reshape(-1).tolist(),
                "device_control": device_output[4].reshape(-1).tolist(),
            }
        )

    host_summary = summarize_samples(host_samples)
    device_summary = summarize_samples(device_samples)
    expected_steps, expected_reason = expected_execution(scenario)
    expected_graph0 = min(expected_steps, scenario.graph_switch_step)
    expected_graph1 = expected_steps - expected_graph0
    expected_control_tail = [
        expected_steps,
        synthetic_token(scenario, expected_steps),
        expected_reason,
        expected_graph0,
        expected_graph1,
    ]
    controls_valid = all(
        row["host_control"][6:11] == expected_control_tail
        and row["device_control"][6:11] == expected_control_tail
        and row["host_control"][11] == row["device_control"][11]
        for row in comparisons
    )
    correct = bool(
        controls_valid and all(row["all_exact"] for row in comparisons)
    )
    return {
        "scenario": scenario.__dict__,
        "expected_steps": expected_steps,
        "expected_finish_reason": expected_reason,
        "correct": correct,
        "host": {
            "feed_calls_per_rep": expected_steps,
            "fetch_calls_per_rep": expected_steps,
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--air-path", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--graph-config", type=Path, required=True)
    parser.add_argument("--func-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 4:
        raise SystemExit("warmup must be non-negative and repetitions >= 4")

    reference = np.load(args.reference)
    initial = (
        reference["input_hidden_table"],
        reference["input_position"],
        reference["input_key_cache"],
        reference["input_value_cache"],
    )
    options = {
        "ge.exec.deviceId": "0",
        "ge.exec.logicalDeviceClusterDeployMode": "SINGLE",
        "ge.exec.logicalDeviceId": "[0:0]",
    }
    df.init(options)
    try:
        host_pairs = [build_host_graph(args, variant) for variant in (0, 1)]
        host_graphs = [item[0] for item in host_pairs]
        host_inputs = [item[1] for item in host_pairs]
        device_graph, device_inputs = build_device_graph(args)
        results = [
            run_scenario(
                scenario,
                args,
                host_graphs,
                host_inputs,
                device_graph,
                device_inputs,
                initial,
            )
            for scenario in SCENARIOS
        ]
    finally:
        df.finalize()

    all_correct = bool(all(result["correct"] for result in results))
    exercised_both_routes = bool(
        any(
            result["comparisons"][0]["device_control"][9] > 0
            and result["comparisons"][0]["device_control"][10] > 0
            for result in results
        )
    )
    result = {
        "schema_version": 1,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "all_correct": all_correct,
        "exercised_both_device_routes": exercised_both_routes,
        "mechanism_gate_pass": bool(all_correct and exercised_both_routes),
        "results": results,
        "claim_boundary": (
            "This is a bounded device-resident control prototype over the real "
            "Qwen layer-0 attention/KV AIR. Tokens and EOS injection are "
            "synthetic, both route keys currently reference the same AIR, the "
            "AIR is not eager-equivalent under the frozen G2d tolerance, and "
            "this is not a full decoder or vLLM integration."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print("BOUNDED_DECODE " + json.dumps(result, ensure_ascii=True), flush=True)
    if not result["mechanism_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
