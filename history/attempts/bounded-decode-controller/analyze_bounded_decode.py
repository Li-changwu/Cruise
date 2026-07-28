#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.result.read_text(encoding="utf-8"))
    rows = []
    for result in raw["results"]:
        host = result["host"]["summary"]
        device = result["device"]["summary"]
        rows.append(
            {
                "scenario": result["scenario"]["name"],
                "executed_steps": result["expected_steps"],
                "finish_reason": (
                    "eos" if result["expected_finish_reason"] == 1 else "max_steps"
                ),
                "host_feed_fetch": result["host"]["feed_calls_per_rep"],
                "device_feed_fetch": result["device"]["feed_calls_per_rep"],
                "host_wall_us_median": host["wall_us_median"],
                "device_wall_us_median": device["wall_us_median"],
                "speedup": result["speedup"],
                "host_cpu_reduction_ratio": result["host_cpu_reduction_ratio"],
                "exact": result["correct"],
            }
        )
    summary = {
        "schema_version": 1,
        "verdict": (
            "BOUNDED_DECODE_CONTROL_PASS_FULL_DECODER_BLOCKED"
            if raw["mechanism_gate_pass"]
            else "BOUNDED_DECODE_CONTROL_FAIL"
        ),
        "mechanism_gate_pass": raw["mechanism_gate_pass"],
        "all_correct": raw["all_correct"],
        "exercised_both_device_routes": raw["exercised_both_device_routes"],
        "rows": rows,
        "claim_boundary": raw["claim_boundary"],
    }
    args.json_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Bounded Device-Resident Decode Control Results",
        "",
        f"Verdict: `{summary['verdict']}`.",
        "",
        "| Scenario | Steps | Stop | Host Feed/Fetch | Device Feed/Fetch | "
        "Host wall median (us) | Device wall median (us) | Speedup | "
        "Host CPU reduction | Exact |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['executed_steps']} | "
            f"{row['finish_reason']} | {row['host_feed_fetch']} | "
            f"{row['device_feed_fetch']} | "
            f"{row['host_wall_us_median']:.1f} | "
            f"{row['device_wall_us_median']:.1f} | {row['speedup']:.2f}x | "
            f"{100 * row['host_cpu_reduction_ratio']:.1f}% | "
            f"{'yes' if row['exact'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The mechanism gate requires exact Host/Device model and control "
            "outputs, correct EOS/max-step termination, both device route keys "
            "to be exercised, and one Host Feed/Fetch for the device route.",
            "",
            "Claim boundary: " + summary["claim_boundary"],
            "",
        ]
    )
    args.markdown_output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
