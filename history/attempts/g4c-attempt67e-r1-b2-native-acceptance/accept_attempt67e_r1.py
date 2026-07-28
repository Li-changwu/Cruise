#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_RESULT_SHA256 = (
    "35d833da9b647a02c8ea58732979eb3d5ad255a174487a1d4787358a92e61858"
)
EXPECTED_NATIVE_OUTPUT_SHA256 = (
    "808c1a3c2a9edff951fa96cbaeda6ea638fbdab063a06f823e7a768132f26530"
)
EXPECTED_CASES = {
    "both-active-heterogeneous",
    "active-plus-empty",
    "finished-plus-active",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_status(path: Path) -> dict[str, int]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        name, value = line.split("\t")
        rows[name] = int(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_raw
    result_path = source / "attempt67e-result.json"
    native_output_path = source / "attempt67e-native-output.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    status = read_status(source / "status.tsv")
    linear = json.loads((source / "linear-launches.json").read_text(encoding="utf-8"))

    metadata_specs = {
        "exactqk": (56, source / "exactqk-launch-metadata.txt"),
        "barrier": (28, source / "barrier-launch-metadata.txt"),
        "materialize": (56, source / "materialize-launch-metadata.txt"),
    }
    metadata = {}
    for name, (expected, path) in metadata_specs.items():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        metadata[name] = {
            "expected_graph_metadata_count": expected,
            "observed_graph_metadata_count": len(lines),
            "pass": len(lines) == expected,
        }

    cases = result.get("cases", [])
    case_names = {case.get("case") for case in cases}
    result_hash = sha256(result_path)
    native_output_hash = sha256(native_output_path)
    required_zero_status = {
        name: status.get(name)
        for name in ("npu7-idle", "native", "compare", "npu7-idle-after")
    }
    idle_files_pass = all(
        "No process in device" in (source / name).read_text(encoding="utf-8")
        for name in ("npu7-processes-before.txt", "npu7-processes-after.txt")
    )
    linear_pass = (
        linear.get("launch_count") == 197
        and linear.get("distinct_op_count") == 197
        and linear.get("all_ops_launched_once") is True
        and linear.get("all_kernels_are_matmulv2") is True
        and linear.get("residual_projections_defused") is True
    )
    cases_pass = (
        len(cases) == 3
        and case_names == EXPECTED_CASES
        and all(case.get("pass") is True for case in cases)
    )
    accepted = (
        result_hash == EXPECTED_RESULT_SHA256
        and native_output_hash == EXPECTED_NATIVE_OUTPUT_SHA256
        and result.get("pass") is True
        and result.get("execution_success") is True
        and cases_pass
        and all(value == 0 for value in required_zero_status.values())
        and idle_files_pass
        and all(item["pass"] for item in metadata.values())
        and linear_pass
    )

    acceptance = {
        "gate": "G4c Attempt 67e-r1 corrected post-hoc acceptance",
        "pass": accepted,
        "source_attempt": "67e",
        "reran_native_graph": False,
        "rationale": (
            "GE emitted kernel metadata once for the built static graph; the three "
            "RunGraph cases must not multiply graph metadata thresholds by three."
        ),
        "source_result_sha256": result_hash,
        "source_native_output_sha256": native_output_hash,
        "source_result_pass": result.get("pass") is True,
        "source_execution_success": result.get("execution_success") is True,
        "source_case_count": len(cases),
        "source_case_names": sorted(case_names),
        "all_source_cases_pass": cases_pass,
        "required_zero_status": required_zero_status,
        "idle_files_pass": idle_files_pass,
        "graph_launch_metadata": metadata,
        "linear_graph_metadata": {
            "expected_graph_metadata_count": 197,
            "observed_graph_metadata_count": linear.get("launch_count"),
            "distinct_op_count": linear.get("distinct_op_count"),
            "all_ops_launched_once": linear.get("all_ops_launched_once"),
            "all_kernels_are_matmulv2": linear.get("all_kernels_are_matmulv2"),
            "residual_projections_defused": linear.get(
                "residual_projections_defused"
            ),
            "pass": linear_pass,
        },
        "claim_boundary": (
            "This closes B=2 single-step native GE semantics only. B=2 resident "
            "Device UDF epochs, independent EOS, B=4, performance, recovery, and "
            "vLLM integration remain open."
        ),
    }
    args.output.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, indent=2), flush=True)
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
