#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "gate": "Attempt 56 general BF16 materialization preflight",
        "shape": [1, 1, 18944],
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "elementwise_exact": args.input.read_bytes() == args.output.read_bytes(),
        "claim_boundary": "Materialization identity only; layer-0 and complete G4a remain untested.",
    }
    result["pass"] = result["elementwise_exact"]
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("BF16_MATERIALIZE_COMPARE " + json.dumps(result), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
