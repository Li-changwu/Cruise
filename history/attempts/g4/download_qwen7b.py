#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    kwargs = {}
    if args.metadata_only:
        kwargs["allow_patterns"] = ["config.json", "model.safetensors.index.json"]
    snapshot = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        max_workers=args.max_workers,
        **kwargs,
    )
    snapshot_path = Path(snapshot)
    result = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "metadata_only": args.metadata_only,
        "snapshot": str(snapshot_path),
        "snapshot_exists": snapshot_path.is_dir(),
        "files": sorted(path.name for path in snapshot_path.iterdir()),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

