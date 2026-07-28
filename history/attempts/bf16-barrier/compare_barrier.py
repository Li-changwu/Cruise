#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps = []
    for step in range(1, 5):
        source = args.input_dir / f"step{step}_qk_scores_bf16.bin"
        output = args.output_dir / source.name
        item = {
            "step": step,
            "input_sha256": sha256(source),
            "output_sha256": sha256(output),
        }
        item["elementwise_exact"] = source.read_bytes() == output.read_bytes()
        steps.append(item)
    result = {
        "gate": "Attempt 54 opaque BF16 materialization barrier",
        "pass": all(item["elementwise_exact"] for item in steps),
        "steps": steps,
        "claim_boundary": "Barrier identity only; Attention and complete decoder remain untested.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("BF16_BARRIER_COMPARE " + json.dumps(result), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

