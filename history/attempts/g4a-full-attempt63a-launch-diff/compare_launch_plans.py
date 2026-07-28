#!/usr/bin/env python3
import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path


START_RE = re.compile(r"Distribute Start, op: (.+?)\s*$")
KERNEL_RE = re.compile(
    r"LaunchKernel: kernel info.*?kernel_name=([^,]+), arg_size=(\d+), "
    r"coreDim=(\d+).*?prefetchCnt1=(\d+)"
)


def parse(path: Path) -> tuple[list[dict], list[str]]:
    launches = []
    missing_kernel = []
    active_op = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            start = START_RE.search(line)
            if start:
                if active_op is not None:
                    missing_kernel.append(active_op)
                active_op = start.group(1)
                continue
            if active_op is None:
                continue
            kernel = KERNEL_RE.search(line)
            if not kernel:
                continue
            launches.append(
                {
                    "op": active_op,
                    "kernel_name": kernel.group(1),
                    "arg_size": int(kernel.group(2)),
                    "core_dim": int(kernel.group(3)),
                    "prefetch_count_1": int(kernel.group(4)),
                }
            )
            active_op = None
    if active_op is not None:
        missing_kernel.append(active_op)
    return launches, missing_kernel


def index_unique(launches: list[dict]) -> tuple[dict[str, dict], dict[str, int]]:
    counts = Counter(item["op"] for item in launches)
    unique = {item["op"]: item for item in launches if counts[item["op"]] == 1}
    duplicates = {name: count for name, count in counts.items() if count != 1}
    return unique, duplicates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline, baseline_missing = parse(args.baseline)
    candidate, candidate_missing = parse(args.candidate)
    baseline_index, baseline_duplicates = index_unique(baseline)
    candidate_index, candidate_duplicates = index_unique(candidate)
    common = sorted(set(baseline_index) & set(candidate_index))
    changed_common = [
        {
            "op": name,
            "baseline": {key: value for key, value in baseline_index[name].items() if key != "op"},
            "candidate": {key: value for key, value in candidate_index[name].items() if key != "op"},
        }
        for name in common
        if baseline_index[name] != candidate_index[name]
    ]

    baseline_names = [item["op"] for item in baseline]
    candidate_names = [item["op"] for item in candidate]
    matcher = difflib.SequenceMatcher(None, baseline_names, candidate_names, autojunk=False)
    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changes.append(
            {
                "tag": tag,
                "baseline_range": [i1, i2],
                "candidate_range": [j1, j2],
                "baseline_ops": baseline_names[i1:i2],
                "candidate_ops": candidate_names[j1:j2],
            }
        )

    valid = not baseline_missing and not candidate_missing
    result = {
        "valid": valid,
        "baseline_launch_count": len(baseline),
        "candidate_launch_count": len(candidate),
        "baseline_distinct_op_count": len(set(baseline_names)),
        "candidate_distinct_op_count": len(set(candidate_names)),
        "baseline_duplicate_ops": baseline_duplicates,
        "candidate_duplicate_ops": candidate_duplicates,
        "baseline_missing_kernel_ops": baseline_missing,
        "candidate_missing_kernel_ops": candidate_missing,
        "common_unique_op_count": len(common),
        "baseline_only_unique_ops": sorted(set(baseline_index) - set(candidate_index)),
        "candidate_only_unique_ops": sorted(set(candidate_index) - set(baseline_index)),
        "changed_common_kernel_count": len(changed_common),
        "changed_common_kernels": changed_common,
        "sequence_equal": baseline_names == candidate_names,
        "sequence_change_count": len(changes),
        "sequence_changes": changes,
        "baseline_kernel_counts": dict(sorted(Counter(item["kernel_name"] for item in baseline).items())),
        "candidate_kernel_counts": dict(sorted(Counter(item["kernel_name"] for item in candidate).items())),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
