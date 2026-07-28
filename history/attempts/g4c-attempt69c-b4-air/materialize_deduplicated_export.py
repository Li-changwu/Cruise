#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def external_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix == "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    base_by_key: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for path in external_files(args.base):
        base_by_key[(path.stat().st_size, sha256(path))].append(path)

    files = []
    hardlinked_bytes = 0
    unique_external_bytes = 0
    for source in sorted(item for item in args.source.iterdir() if item.is_file()):
        destination = args.output / source.name
        digest = sha256(source)
        entry = {
            "name": source.name,
            "bytes": source.stat().st_size,
            "sha256": digest,
            "external": source.suffix == "",
            "hardlinked_to": None,
        }
        if source.suffix == "":
            matches = base_by_key.get((source.stat().st_size, digest), [])
            if matches:
                os.link(matches[0], destination)
                entry["hardlinked_to"] = str(matches[0])
                hardlinked_bytes += source.stat().st_size
            else:
                shutil.copy2(source, destination)
                unique_external_bytes += source.stat().st_size
        else:
            shutil.copy2(source, destination)
        if sha256(destination) != digest:
            raise RuntimeError(f"materialized hash mismatch: {destination}")
        if entry["hardlinked_to"] is not None:
            if destination.stat().st_ino != Path(entry["hardlinked_to"]).stat().st_ino:
                raise RuntimeError(f"hardlink inode mismatch: {destination}")
        files.append(entry)

    external = [item for item in files if item["external"]]
    result = {
        "valid": True,
        "source": str(args.source),
        "base": str(args.base),
        "output": str(args.output),
        "file_count": len(files),
        "external_file_count": len(external),
        "hardlinked_external_file_count": sum(
            item["hardlinked_to"] is not None for item in external
        ),
        "unique_external_file_count": sum(
            item["hardlinked_to"] is None for item in external
        ),
        "external_bytes": sum(item["bytes"] for item in external),
        "hardlinked_external_bytes": hardlinked_bytes,
        "unique_external_bytes": unique_external_bytes,
        "files": files,
    }
    manifest = args.output / "dedup-manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
