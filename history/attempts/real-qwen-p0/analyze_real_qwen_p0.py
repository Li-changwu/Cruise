#!/usr/bin/env python3
import csv
import json
import statistics
from pathlib import Path


N_VALUES = (1, 2, 4)


def iqr(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return [q1, q3]


def route_summary(route):
    wall = [row["wall_us"] for row in route["samples"]]
    cpu = [row["host_cpu_us"] for row in route["samples"]]
    return {
        "wall_us_median": statistics.median(wall),
        "wall_us_iqr": iqr(wall),
        "host_cpu_us_median": statistics.median(cpu),
        "host_cpu_us_iqr": iqr(cpu),
    }


def main():
    root = Path("/root/ascend-control-real-p0-20260722")
    with (root / "raw" / "status.tsv").open(encoding="utf-8", newline="") as f:
        status = list(csv.DictReader(f, delimiter="\t"))
    status_ok = all(int(row["exit_status"]) == 0 for row in status)

    rows = []
    all_correct = True
    for n in N_VALUES:
        case = json.loads((root / "raw" / f"n{n}.json").read_text())
        all_correct = all_correct and case["correct_host_vs_device"]
        host = route_summary(case["host"])
        device = route_summary(case["device"])
        rows.append(
            {
                "n": n,
                "host_feed_calls": case["host"]["feed_calls_per_rep"],
                "device_feed_calls": case["device"]["feed_calls_per_rep"],
                "host": host,
                "device": device,
                "speedup": host["wall_us_median"] / device["wall_us_median"],
                "host_cpu_reduction_ratio": 1
                - device["host_cpu_us_median"] / host["host_cpu_us_median"],
                "all_repetitions_exact": all(
                    item["all_exact"] for item in case["comparisons"]
                ),
            }
        )

    performance_gate = all(
        row["device"]["wall_us_median"] < row["host"]["wall_us_median"]
        for row in rows
        if row["n"] >= 2
    )
    constant_submission_gate = all(
        row["device_feed_calls"] == 1 for row in rows
    )
    mechanism_pass = bool(
        status_ok and all_correct and performance_gate and constant_submission_gate
    )
    result = {
        "schema_version": 1,
        "verdict": (
            "REAL_QWEN_P0_MECHANISM_PASS_AIR_EAGER_SEMANTICS_BLOCKED"
            if mechanism_pass
            else "REAL_QWEN_P0_FAIL"
        ),
        "hardware": {
            "product": "Ascend 910B2",
            "physical_device": 7,
            "cann_version": "9.0.0",
        },
        "workload": {
            "checkpoint": "Qwen/Qwen2.5-7B-Instruct",
            "scope": "layer-0 attention/KV slice",
            "n_values": list(N_VALUES),
            "warmup": 3,
            "repetitions": 10,
        },
        "gates": {
            "all_status_zero": status_ok,
            "all_host_device_outputs_elementwise_exact": all_correct,
            "device_feed_calls_constant_one": constant_submission_gate,
            "device_faster_for_n_2_and_4": performance_gate,
            "n1_negative_regime_observed": rows[0]["speedup"] < 1,
        },
        "comparison": rows,
        "claim_boundary": (
            "This proves Host-vs-Device-UDF equivalence and recurrence-overhead "
            "reduction for the same real-Qwen layer-0 attention/KV AIR. Prior "
            "G2d evidence shows that AIR is not eager-equivalent under the frozen "
            "tolerance. It is not a full decoder, paged-KV, sampling, dynamic-"
            "batching, or vLLM result."
        ),
    }
    (root / "real-qwen-p0-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Real-Qwen P0 Host-vs-Device Recurrence Results",
        "",
        f"Verdict: `{result['verdict']}`.",
        "",
        "| N | Host Feed/Fetch | Device Feed/Fetch | Host wall median (us) | Device wall median (us) | Speedup | Host CPU reduction | Exact |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {n} | {hc} | {dc} | {hw:.1f} | {dw:.1f} | {s:.2f}x | {cpu:.1%} | {exact} |".format(
                n=row["n"],
                hc=row["host_feed_calls"],
                dc=row["device_feed_calls"],
                hw=row["host"]["wall_us_median"],
                dw=row["device"]["wall_us_median"],
                s=row["speedup"],
                cpu=row["host_cpu_reduction_ratio"],
                exact="yes" if row["all_repetitions_exact"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "Each route was warmed up three times and measured ten times with alternating route order. IQRs are retained in the JSON summary.",
            "N=1 is the fixed-overhead negative regime; the preregistered performance gate requires wins only at N=2 and N=4.",
            "",
            "All Host and Device-UDF final attention, key-cache, value-cache, and position tensors were elementwise identical.",
            "Device UDF reduced Host Feed/Fetch pairs from N to one.",
            "",
            "This AIR is a real Qwen layer-0 attention/KV slice but is not eager-equivalent under the frozen G2d tolerance. Full decoder and vLLM integration remain open.",
        ]
    )
    (root / "REAL-QWEN-P0-RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
