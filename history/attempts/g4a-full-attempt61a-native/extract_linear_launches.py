#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=197)
    args = parser.parse_args()

    launches = []
    active_op = None
    with args.log.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            start = re.search(
                r"Distribute Start, op: ((?:Qkv)?LinearTransposeX2_[0-9]+)", line
            )
            if start:
                active_op = start.group(1)
                continue
            if active_op is None:
                continue
            if "LaunchKernel: kernel info" in line:
                match = re.search(
                    r"kernel_name=([^,]+), arg_size=(\d+), coreDim=(\d+).*?prefetchCnt1=(\d+)",
                    line,
                )
                if not match:
                    raise ValueError(f"failed to parse launch metadata for {active_op}")
                launches.append(
                    {
                        "op": active_op,
                        "kernel_name": match.group(1),
                        "arg_size": int(match.group(2)),
                        "core_dim": int(match.group(3)),
                        "prefetch_count_1": int(match.group(4)),
                    }
                )
                active_op = None

    op_counts = Counter(item["op"] for item in launches)
    kernel_counts = Counter(item["kernel_name"] for item in launches)
    valid = (
        len(launches) == args.expected
        and len(op_counts) == args.expected
        and set(op_counts.values()) == {1}
        and all("matmulv2" in item["kernel_name"].lower() for item in launches)
    )
    result = {
        "valid": valid,
        "expected_launch_count": args.expected,
        "launch_count": len(launches),
        "distinct_op_count": len(op_counts),
        "all_ops_launched_once": bool(op_counts) and set(op_counts.values()) == {1},
        "all_kernels_are_matmulv2": all(
            "matmulv2" in item["kernel_name"].lower() for item in launches
        ),
        "op_launch_counts": dict(sorted(op_counts.items())),
        "kernel_launch_counts": dict(sorted(kernel_counts.items())),
        "launches": launches,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not valid:
        raise SystemExit(91)


if __name__ == "__main__":
    main()
