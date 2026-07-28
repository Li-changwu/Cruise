#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_HASHES = {
    "qwen_b4_decoder_step_attempt69c.air": "de4a7bf439337970b343eb1fa91c3dd326545e81a88c94680c355234f96044bb",
    "dynamo.pbtxt": "1594d0c232de8ed8c0c983908dc82670952ee8ef4b2d796fa564292027ff6697",
    "abi.json": "045b3bdf7537921789f7cd8489b068455c6d2c5397906de2dc966e29127ade6e",
    "graph-inspection.json": "776c89b7926b6c6e213be9d01eba2606587dabea924a27f80deee953d9d310b2",
    "export-result.json": "246e956681d3c6090b764b269c35687447a7c5cfb297422eb8d65521dee4faaa",
    "dedup-manifest.json": "15f9bdb4decb0fa3680b0c81e7cb16984b54a6aae1b8792f2c9cfc77d83cedba",
}
EXPECTED_STATUS_SHA256 = "2f8c0eaa386993aab799764585111b13d21d695798402d9cd0bc1e1613511af5"
EXPECTED_FEED_ORDER = [
    "token_id",
    "position",
    "sequence_length",
    "key_cache",
    "slot_mapping",
    "active_mask",
    "block_table",
    "value_cache",
    "explicit_tiling",
]
EXPECTED_OP_COUNTS = {
    "MatMul": 0,
    "MatMulV2": 197,
    "ExactQk": 112,
    "Bf16Barrier": 28,
    "Bf16Materialize": 56,
    "BatchMatMul": 28,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_status(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {name: int(value) for name, value in (line.split("\t") for line in lines[1:])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-raw", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    status_path = args.source_raw / "status.tsv"
    status = read_status(status_path)
    status_pass = bool(status) and all(value == 0 for value in status.values())
    observed_hashes = {
        name: sha256(args.export_dir / name) for name in EXPECTED_HASHES
    }
    frozen_hashes_pass = observed_hashes == EXPECTED_HASHES
    status_hash = sha256(status_path)

    export_result = json.loads(
        (args.export_dir / "export-result.json").read_text(encoding="utf-8")
    )
    abi = json.loads((args.export_dir / "abi.json").read_text(encoding="utf-8"))
    graph = json.loads(
        (args.export_dir / "graph-inspection.json").read_text(encoding="utf-8")
    )
    dedup = json.loads(
        (args.export_dir / "dedup-manifest.json").read_text(encoding="utf-8")
    )

    current_file_hashes_pass = True
    hardlink_inodes_pass = True
    checked_files = 0
    for item in dedup["files"]:
        current = args.export_dir / item["name"]
        if not current.is_file() or sha256(current) != item["sha256"]:
            current_file_hashes_pass = False
        if item["hardlinked_to"] is not None:
            base = Path(item["hardlinked_to"])
            if not base.is_file() or current.stat().st_ino != base.stat().st_ino:
                hardlink_inodes_pass = False
        checked_files += 1

    dedup_summary_pass = (
        dedup.get("valid") is True
        and dedup.get("external_file_count") == 342
        and dedup.get("hardlinked_external_file_count") == 341
        and dedup.get("unique_external_file_count") == 1
        and dedup.get("hardlinked_external_bytes") == 15231233312
        and dedup.get("unique_external_bytes") == 4096
    )
    export_pass = (
        export_result.get("execution_success") is True
        and export_result.get("air_sha256") == EXPECTED_HASHES["qwen_b4_decoder_step_attempt69c.air"]
        and export_result.get("reference_sha256")
        == "a7d65e455a77a561352a8f3796d94ec86e1e429ebe942feacd5b14013123fdd8"
        and export_result.get("external_file_count") == 342
    )
    abi_pass = abi.get("valid") is True and abi.get("native_feed_order") == EXPECTED_FEED_ORDER
    graph_pass = (
        graph.get("valid") is True
        and graph.get("observed_op_counts") == EXPECTED_OP_COUNTS
        and graph.get("active_mask_data_count") == 1
        and graph.get("slot_mapping_data_count") == 1
    )
    accepted = (
        status_pass
        and status_hash == EXPECTED_STATUS_SHA256
        and frozen_hashes_pass
        and export_pass
        and abi_pass
        and graph_pass
        and dedup_summary_pass
        and current_file_hashes_pass
        and hardlink_inodes_pass
    )
    result = {
        "gate": "G4c Attempt 69c-r1 corrected B=4 AIR post-hoc acceptance",
        "pass": accepted,
        "reran_export": False,
        "source_status": status,
        "source_status_sha256": status_hash,
        "frozen_hashes": observed_hashes,
        "export_pass": export_pass,
        "abi_pass": abi_pass,
        "native_feed_order": abi.get("native_feed_order"),
        "graph_pass": graph_pass,
        "observed_op_counts": graph.get("observed_op_counts"),
        "dedup_summary_pass": dedup_summary_pass,
        "checked_manifest_files": checked_files,
        "current_file_hashes_pass": current_file_hashes_pass,
        "hardlink_inodes_pass": hardlink_inodes_pass,
        "reporting_defect": (
            "Attempt 69c's final stdout summary requested a removed JSON key. "
            "All recorded gate statuses were zero; this acceptance uses the "
            "actual active_mask_data_count and slot_mapping_data_count fields."
        ),
        "claim_boundary": (
            "This closes B=4 AIR export and structural integrity only. Native "
            "GE numerics, resident epochs, performance, recovery, and vLLM "
            "integration remain open."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
