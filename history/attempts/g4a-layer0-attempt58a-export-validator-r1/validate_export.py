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


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def status(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {name: int(value) for name, value in (line.split("\t") for line in lines[1:])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-raw", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--graph-inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = status(args.original_raw / "status.tsv")
    export_result = load(args.export_dir / "export-result.json")
    eager = load(args.export_dir / "eager-comparison.json")
    abi = load(args.export_dir / "abi.json")
    graph = load(args.graph_inspection)
    air = args.export_dir / "qwen_layer0_boundary_attempt58a.air"
    pbtxt = args.export_dir / "dynamo.pbtxt"
    reference = args.export_dir / "attempt58a-eager-reference.npz"
    original_expected = {
        "npu7-idle": 0,
        "export": 0,
        "eager-comparison": 0,
        "abi": 0,
        "graph-inspection": 91,
    }
    valid = (
        original == original_expected
        and export_result["pass"]
        and export_result["eager_pass"]
        and eager["valid"]
        and abi["valid"]
        and graph["valid"]
        and air.is_file()
        and pbtxt.is_file()
        and reference.is_file()
    )
    result = {
        "valid": valid,
        "classification": "original verifier false negative" if valid else "unresolved",
        "original_status": original,
        "original_expected_status": original_expected,
        "export_result": export_result,
        "eager_comparison": eager,
        "abi_valid": abi["valid"],
        "graph_inspection": graph,
        "artifacts": {
            "air": {"bytes": air.stat().st_size, "sha256": sha256(air)},
            "pbtxt": {"bytes": pbtxt.stat().st_size, "sha256": sha256(pbtxt)},
            "eager_reference": {
                "bytes": reference.stat().st_size,
                "sha256": sha256(reference),
            },
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not valid:
        raise SystemExit(92)


if __name__ == "__main__":
    main()
