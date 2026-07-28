#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    launches = []
    in_gate = False
    with args.log.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "Distribute Start, op: GateMatMulV2TransposeX2" in line:
                in_gate = True
                continue
            if not in_gate:
                continue
            if "LaunchKernel: kernel info" in line:
                match = re.search(
                    r"kernel_name=([^,]+), arg_size=(\d+), coreDim=(\d+).*?prefetchCnt1=(\d+)",
                    line,
                )
                if not match:
                    raise ValueError("failed to parse gate launch metadata")
                launches.append(
                    {
                        "kernel_name": match.group(1),
                        "arg_size": int(match.group(2)),
                        "core_dim": int(match.group(3)),
                        "prefetch_count_1": int(match.group(4)),
                        "raw": line.strip(),
                    }
                )
            if "Distribute Success, op: GateMatMulV2TransposeX2" in line:
                in_gate = False

    valid = (
        len(launches) == 1
        and "matmulv2" in launches[0]["kernel_name"].lower()
        and launches[0]["core_dim"] == 19
    )
    result = {
        "valid": valid,
        "launch_count": len(launches),
        "launches": launches,
        "eager_control": {
            "kernel_name": "MatMulV2_ND_ND_FP16_FP16_false_true_all_98499",
            "arg_size": 200,
            "core_dim": 19,
            "prefetch_count_1": 3,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not valid:
        raise SystemExit(91)


if __name__ == "__main__":
    main()
