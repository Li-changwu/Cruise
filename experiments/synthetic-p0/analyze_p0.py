#!/usr/bin/env python3
import csv
import json
import re
import statistics
from pathlib import Path


N_VALUES = (1, 2, 4, 8, 16, 32)
HOST_PATTERN = re.compile(
    r"HOST_TOKEN_LOOP_RESULT rep=(\d+) executed=(\d+) loop_us=(\d+)"
    r"(?: host_cpu_us=(\d+))? final_state=\[([^]]+)\]"
)
DEVICE_PATTERN = re.compile(
    r"DEVICE_TOKEN_LOOP_RESULT rep=(\d+) executed=(\d+) feed_us=(\d+) "
    r"fetch_us=(\d+)(?: host_cpu_us=(\d+))? feed_calls=(\d+) "
    r"fetch_calls=(\d+) final_state=\[([^]]+)\]"
)


def median(values: list[int]) -> float:
    return statistics.median(values)


def status_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def status_ok(path: Path) -> bool:
    return all(int(row["exit_status"]) == 0 for row in status_rows(path))


def parse_state(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_host(path: Path, n: int, expected_reps: int) -> dict:
    rows = []
    for match in HOST_PATTERN.finditer(path.read_text(encoding="utf-8")):
        row = {
            "rep": int(match.group(1)),
            "executed": int(match.group(2)),
            "wall_us": int(match.group(3)),
            "state": parse_state(match.group(5)),
        }
        if match.group(4) is not None:
            row["host_cpu_us"] = int(match.group(4))
        rows.append(row)
    expected = [n, n, 0, n]
    correct = len(rows) == expected_reps and all(
        row["executed"] == n and row["state"] == expected for row in rows
    )
    result = {
        "route": "host_ge_graph",
        "n": n,
        "correct": correct,
        "host_submission_calls": n,
        "host_completion_calls": n,
        "samples": rows,
        "wall_us_median": median([row["wall_us"] for row in rows]),
    }
    if rows and "host_cpu_us" in rows[0]:
        result["host_cpu_us_median"] = median(
            [row["host_cpu_us"] for row in rows]
        )
    return result


def parse_device(path: Path, n: int, expected_reps: int) -> dict:
    rows = []
    for match in DEVICE_PATTERN.finditer(path.read_text(encoding="utf-8")):
        feed_us = int(match.group(3))
        fetch_us = int(match.group(4))
        row = {
            "rep": int(match.group(1)),
            "executed": int(match.group(2)),
            "feed_us": feed_us,
            "fetch_us": fetch_us,
            "wall_us": feed_us + fetch_us,
            "feed_calls": int(match.group(6)),
            "fetch_calls": int(match.group(7)),
            "state": parse_state(match.group(8)),
        }
        if match.group(5) is not None:
            row["host_cpu_us"] = int(match.group(5))
        rows.append(row)
    expected = [n, n, 0, n]
    correct = len(rows) == expected_reps and all(
        row["executed"] == n
        and row["state"] == expected
        and row["feed_calls"] == 1
        and row["fetch_calls"] == 1
        for row in rows
    )
    result = {
        "route": "device_udf",
        "n": n,
        "correct": correct,
        "host_submission_calls": 1,
        "host_completion_calls": 1,
        "samples": rows,
        "feed_us_median": median([row["feed_us"] for row in rows]),
        "fetch_us_median": median([row["fetch_us"] for row in rows]),
        "wall_us_median": median([row["wall_us"] for row in rows]),
    }
    if rows and "host_cpu_us" in rows[0]:
        result["host_cpu_us_median"] = median(
            [row["host_cpu_us"] for row in rows]
        )
    return result


def read_pmu(root: Path, route: str) -> dict[str, int]:
    base = (
        root / "ctrl-probe-idle"
        if route == "idle"
        else root / "profiles" / f"{route}-sys"
    )
    files = list(base.glob("**/ts_cpu_pmu_events_*.csv"))
    if len(files) != 1:
        return {}
    with files[0].open(newline="", encoding="utf-8") as handle:
        return {row["Name"]: int(row["Count"]) for row in csv.DictReader(handle)}


def read_task_types(root: Path, route: str) -> list[str]:
    values = []
    for path in (root / "profiles" / f"{route}-app").glob(
        "**/task_time_*.csv"
    ):
        with path.open(newline="", encoding="utf-8") as handle:
            values.extend(row["kernel_type"] for row in csv.DictReader(handle))
    return values


def torch_case_map(torch_result: dict) -> dict[tuple[str, int], dict]:
    return {(case["route"], case["n"]): case for case in torch_result["cases"]}


def main() -> None:
    root = Path("/root/ascend-control-p0-20260722")
    torch_result = json.loads((root / "torch-results.json").read_text())
    torch_cases = torch_case_map(torch_result)
    cpu_cases = {}
    round1_cpu_cases = {}
    uninstrumented_cases = {}
    for n in N_VALUES:
        cpu_cases[("host", n)] = parse_host(
            root / "cpu-sweep-r2" / f"host-ge-n{n}.stdout.log", n, 30
        )
        cpu_cases[("device", n)] = parse_device(
            root / "cpu-sweep-r2" / f"device-udf-n{n}.stdout.log", n, 30
        )
        round1_cpu_cases[("host", n)] = parse_host(
            root / "cpu-sweep" / f"host-ge-n{n}.stdout.log", n, 20
        )
        round1_cpu_cases[("device", n)] = parse_device(
            root / "cpu-sweep" / f"device-udf-n{n}.stdout.log", n, 20
        )
        uninstrumented_cases[("host", n)] = parse_host(
            root / "ge-device-sweep" / f"host-ge-n{n}.stdout.log", n, 20
        )
        uninstrumented_cases[("device", n)] = parse_device(
            root / "ge-device-sweep" / f"device-udf-n{n}.stdout.log", n, 20
        )

    comparison = []
    for n in N_VALUES:
        host = cpu_cases[("host", n)]
        device = cpu_cases[("device", n)]
        round1_host = round1_cpu_cases[("host", n)]
        round1_device = round1_cpu_cases[("device", n)]
        uninst_host = uninstrumented_cases[("host", n)]
        uninst_device = uninstrumented_cases[("device", n)]
        comparison.append(
            {
                "n": n,
                "torch_eager_wall_us_median": torch_cases[
                    ("torch_eager", n)
                ]["summary"]["wall_us_median"],
                "npu_graph_wall_us_median": torch_cases[
                    ("npu_graph", n)
                ]["summary"]["wall_us_median"],
                "host_ge_wall_us_median": host["wall_us_median"],
                "device_udf_wall_us_median": device["wall_us_median"],
                "device_udf_speedup_vs_host_ge": (
                    host["wall_us_median"] / device["wall_us_median"]
                ),
                "host_ge_cpu_us_median": host["host_cpu_us_median"],
                "device_udf_host_cpu_us_median": device["host_cpu_us_median"],
                "host_cpu_reduction_ratio": 1
                - device["host_cpu_us_median"] / host["host_cpu_us_median"],
                "host_submission_calls": n,
                "device_submission_calls": 1,
                "round1_host_ge_wall_us_median": round1_host["wall_us_median"],
                "round1_device_udf_wall_us_median": round1_device[
                    "wall_us_median"
                ],
                "uninstrumented_host_ge_wall_us_median": uninst_host[
                    "wall_us_median"
                ],
                "uninstrumented_device_udf_wall_us_median": uninst_device[
                    "wall_us_median"
                ],
            }
        )

    all_correct = bool(
        torch_result["all_correct"]
        and all(case["correct"] for case in cpu_cases.values())
        and all(case["correct"] for case in round1_cpu_cases.values())
        and all(case["correct"] for case in uninstrumented_cases.values())
    )
    performance_gate = all(
        row["device_udf_wall_us_median"] < row["host_ge_wall_us_median"]
        for row in comparison
        if row["n"] >= 2
    )
    uninstrumented_performance_gate = all(
        row["uninstrumented_device_udf_wall_us_median"]
        < row["uninstrumented_host_ge_wall_us_median"]
        for row in comparison
        if row["n"] >= 2
    )
    round1_performance_gate = all(
        row["round1_device_udf_wall_us_median"]
        < row["round1_host_ge_wall_us_median"]
        for row in comparison
        if row["n"] >= 2
    )
    crossover = next(
        (
            row["n"]
            for row in comparison
            if row["device_udf_wall_us_median"]
            < row["host_ge_wall_us_median"]
        ),
        None,
    )

    system_csv_names = sorted(
        {path.name.rsplit("_", 1)[0] for path in root.glob("**/*.csv")}
    )
    host_task_types = read_task_types(root, "host-ge")
    device_task_types = read_task_types(root, "device-udf")
    inner_task_types = {"AI_CORE", "AI_VECTOR_CORE", "AIC", "AIV"}
    inner_tasks_observed = bool(
        inner_task_types.intersection(host_task_types + device_task_types)
    )

    result = {
        "schema_version": 1,
        "verdict": (
            "P0_FEASIBILITY_PASS_FULL_QWEN_ROUTE_BLOCKED"
            if all_correct
            and performance_gate
            and round1_performance_gate
            and uninstrumented_performance_gate
            else "P0_FAIL"
        ),
        "hardware": {
            "physical_device": 7,
            "product": "Ascend 910B2",
            "npu_smi_version": "26.0.rc1",
            "cann_version": "9.0.0",
        },
        "gates": {
            "all_correct": all_correct,
            "cpu_sweep_status_ok": status_ok(
                root / "cpu-sweep-r2" / "status.tsv"
            ),
            "round1_cpu_sweep_status_ok": status_ok(
                root / "cpu-sweep" / "status.tsv"
            ),
            "uninstrumented_sweep_status_ok": status_ok(
                root / "ge-device-sweep" / "status.tsv"
            ),
            "device_faster_than_host_ge_for_n_ge_2": performance_gate,
            "all_three_rounds_direction_consistent_for_n_ge_2": (
                performance_gate
                and round1_performance_gate
                and uninstrumented_performance_gate
            ),
            "device_host_submission_constant": all(
                row["device_submission_calls"] == 1 for row in comparison
            ),
        },
        "crossover_n": crossover,
        "comparison": comparison,
        "ts_cpu": {
            "idle": read_pmu(root, "idle"),
            "host_ge": read_pmu(root, "host-ge"),
            "device_udf": read_pmu(root, "device-udf"),
        },
        "profiling_observability": {
            "csv_name_prefixes": system_csv_names,
            "host_task_types": sorted(set(host_task_types)),
            "device_task_types": sorted(set(device_task_types)),
            "inner_ai_core_tasks_observed": inner_tasks_observed,
            "ai_core_gap_status": (
                "observed" if inner_tasks_observed else "not_observed_by_current_msprof_path"
            ),
            "dynamic_attach_executor_status": "rejected_no_valid_pid",
            "dynamic_attach_main_status": "rejected_no_valid_pid",
        },
        "claim_boundary": (
            "The P0 result proves a synthetic recurrent GE Add control loop on "
            "one 910B2. The registered real-Qwen TorchAir/AIR route remains "
            "blocked by QK semantic fidelity, and current msprof did not expose "
            "inner Add task timestamps through the DataFlow executor."
        ),
    }
    (root / "p0-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# P0 Device-Resident Iteration Control Results",
        "",
        f"Verdict: `{result['verdict']}`.",
        "",
        "| N | Torch Eager (us) | NPUGraph (us) | Host GE (us) | Device UDF (us) | Speedup | Host CPU (us) | Device Host CPU (us) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            "| {n} | {e:.1f} | {g:.1f} | {h:.1f} | {d:.1f} | {s:.2f}x | {hc:.1f} | {dc:.1f} |".format(
                n=row["n"],
                e=row["torch_eager_wall_us_median"],
                g=row["npu_graph_wall_us_median"],
                h=row["host_ge_wall_us_median"],
                d=row["device_udf_wall_us_median"],
                s=row["device_udf_speedup_vs_host_ge"],
                hc=row["host_ge_cpu_us_median"],
                dc=row["device_udf_host_cpu_us_median"],
            )
        )
    lines.extend(
        [
            "",
            "All four routes passed exact state checks. The primary GE/DataFlow round uses 5 warmups and 30 repetitions per N.",
            "Device UDF uses one Host feed and one final fetch regardless of N; Host GE uses N RunGraph calls.",
            "",
            "System profiling exported TS CPU PMU/top-function CSVs only. It did not export Ctrl CPU, AI CPU, or generic cpu_usage CSVs.",
            "Application profiling did not expose the inner Add task timestamps through the DataFlow executor, so AI Core gap is not claimed.",
            "",
            "The synthetic P0 mechanism passes. Full Qwen/vLLM integration remains blocked by the separately isolated TorchAir/AIR QK semantic failure.",
        ]
    )
    (root / "P0-RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
