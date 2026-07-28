#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_status(path: Path) -> dict[str, int]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows or rows[0] != "case\texit_status":
        raise ValueError("invalid status header")
    result = {}
    for row in rows[1:]:
        name, value = row.split("\t")
        result[name] = int(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--relocation", type=Path, required=True)
    parser.add_argument("--linear", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--air", type=Path, required=True)
    parser.add_argument("--exact-count", type=int, required=True)
    parser.add_argument("--barrier-count", type=int, required=True)
    parser.add_argument("--materialize-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    relocation = json.loads(args.relocation.read_text(encoding="utf-8"))
    linear = json.loads(args.linear.read_text(encoding="utf-8"))
    status = read_status(args.status)
    expected_statuses = {
        "npu7-idle",
        "prepare",
        "compile",
        "native",
        "compare",
        "exactqk-launch",
        "barrier-launch",
        "materialize-launch",
        "linear-launches",
        "npu7-idle-after",
    }
    status_pass = set(status) == expected_statuses and all(
        value == 0 for value in status.values()
    )
    launch_counts = {
        "ExactQk": args.exact_count,
        "Bf16Barrier": args.barrier_count,
        "Bf16Materialize": args.materialize_count,
        "decoder_linear": int(linear.get("launch_count", -1)),
    }
    launch_pass = launch_counts == {
        "ExactQk": 112,
        "Bf16Barrier": 28,
        "Bf16Materialize": 56,
        "decoder_linear": 197,
    }
    result = {
        "gate": "G4c Attempt 69d-r1 B=4 native GE acceptance",
        "pass": bool(
            comparison.get("pass")
            and relocation.get("pass")
            and linear.get("valid")
            and status_pass
            and launch_pass
        ),
        "comparison_pass": bool(comparison.get("pass")),
        "relocation_pass": bool(relocation.get("pass")),
        "linear_metadata_pass": bool(linear.get("valid")),
        "status_pass": status_pass,
        "status": status,
        "launch_counts": launch_counts,
        "air_sha256": sha256(args.air),
        "comparison_sha256": sha256(args.comparison),
        "relocation_sha256": sha256(args.relocation),
        "claim_boundary": (
            "This closes B=4 single-step native GE semantics only. B=4 "
            "resident epochs, performance, recovery, and vLLM integration "
            "remain open."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
