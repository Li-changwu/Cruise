#!/usr/bin/env python3
"""Promote a mutable GE external-weight view into an XFS asset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path


SCHEMA = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def canonical_content(
    canonical: Path, reuse_bundles: list[Path]
) -> dict[tuple[int, str], Path]:
    content: dict[tuple[int, str], Path] = {}
    for bundle, external_only in (
        (canonical, True),
        *((bundle, False) for bundle in reuse_bundles),
    ):
        manifest = load_json(bundle / "dedup-manifest.json")
        files = manifest.get("files")
        if not isinstance(files, list) or not manifest.get("valid"):
            raise ValueError(f"invalid reusable bundle manifest: {bundle}")
        for raw in files:
            if not isinstance(raw, dict) or (external_only and not raw.get("external")):
                continue
            name = raw.get("name")
            size = raw.get("bytes")
            digest = raw.get("sha256")
            if not isinstance(name, str) or not isinstance(size, int) or not isinstance(digest, str):
                raise ValueError("invalid canonical file record")
            path = bundle / name
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"canonical file mismatch: {path}")
            content.setdefault((size, digest), path)
    return content


def source_records(
    source: Path, generation_gap_seconds: float, generation: str
) -> tuple[list[dict[str, object]], int, int]:
    meta = load_json(source / "meta.json")
    mapping = meta.get("hash_to_weight_file")
    offsets = meta.get("hash_to_weight_offset")
    if not isinstance(mapping, dict) or not isinstance(offsets, dict):
        raise ValueError("GE meta.json has an invalid schema")
    records: list[dict[str, object]] = []
    for weight_hash, raw_path in sorted(mapping.items()):
        if not isinstance(weight_hash, str) or not isinstance(raw_path, str):
            raise ValueError("GE meta mapping must contain string keys and paths")
        path = Path(raw_path).resolve(strict=True)
        if path.parent != source or not path.name.startswith("weight_") or not path.is_file():
            raise ValueError(f"GE meta path escapes source: {path}")
        file_stat = path.stat()
        records.append(
            {
                "hash": weight_hash,
                "name": path.name,
                "source": str(path),
                "bytes": file_stat.st_size,
                "mtime": file_stat.st_mtime,
            }
        )
    if len({record["name"] for record in records}) != len(records):
        raise ValueError("GE meta contains duplicate weight filenames")
    by_mtime = sorted(records, key=lambda record: float(record["mtime"]))
    generations: list[list[dict[str, object]]] = []
    generation_start = 0
    for index in range(1, len(by_mtime)):
        if (
            float(by_mtime[index]["mtime"])
            - float(by_mtime[index - 1]["mtime"])
            > generation_gap_seconds
        ):
            generations.append(by_mtime[generation_start:index])
            generation_start = index
    generations.append(by_mtime[generation_start:])
    if generation == "all":
        generation_index = -1
        selected = by_mtime
    else:
        generation_index = 0 if generation == "oldest" else len(generations) - 1
        selected = generations[generation_index]
    if not selected:
        raise ValueError("GE meta.json references no weight files")
    for record in selected:
        record["sha256"] = sha256(Path(str(record["source"])))
    return (
        sorted(selected, key=lambda record: str(record["hash"])),
        generation_index,
        len(generations),
    )


def identity(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (
                f"{record['hash']}\t{record['name']}\t{record['bytes']}\t"
                f"{record['sha256']}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def ensure_runtime_traversal(assets_root: Path) -> None:
    """Allow the GE compiler subprocess to traverse user-owned asset parents."""
    owner = assets_root.stat().st_uid
    required = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    for directory in (assets_root, *assets_root.parents):
        info = directory.stat()
        if info.st_uid != owner:
            break
        mode = stat.S_IMODE(info.st_mode)
        if mode & required != required:
            os.chmod(directory, mode | required)


def validate_promoted_bundle(
    output: Path,
    source: Path,
    records: list[dict[str, object]],
    expected_identity: str,
    require_source_hardlinks: bool,
    canonical_by_content: dict[tuple[int, str], Path],
) -> None:
    manifest = load_json(output / "dedup-manifest.json")
    meta = load_json(output / "meta.json")
    mapping = meta.get("hash_to_weight_file")
    if (
        not manifest.get("valid")
        or manifest.get("identity_sha256") != expected_identity
        or not isinstance(mapping, dict)
    ):
        raise ValueError(f"invalid promoted external-weight bundle: {output}")
    expected_hashes = {str(record["hash"]) for record in records}
    if set(mapping) != expected_hashes:
        raise ValueError("promoted external-weight key set mismatch")
    for record in records:
        weight_hash = str(record["hash"])
        destination = Path(str(mapping[weight_hash])).resolve(strict=True)
        if (
            destination.parent != output
            or destination.stat().st_size != int(record["bytes"])
        ):
            raise ValueError(f"promoted external-weight file mismatch: {destination}")
        if require_source_hardlinks:
            key = (int(record["bytes"]), str(record["sha256"]))
            original = canonical_by_content.get(
                key, Path(str(record["source"])).resolve(strict=True)
            )
            if not os.path.samestat(original.stat(), destination.stat()):
                raise ValueError(
                    f"promoted external weight is not hard-linked: {destination}"
                )
        elif sha256(destination) != str(record["sha256"]):
            raise ValueError(
                f"promoted external-weight digest mismatch: {destination}"
            )


def promote(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve(strict=True)
    canonical = Path(args.canonical).resolve(strict=True)
    assets_root = Path(args.assets_root).resolve(strict=True)
    runtime_root = Path(args.runtime_root).resolve(strict=True)
    reuse_bundles = [Path(path).resolve(strict=True) for path in args.reuse_bundle]
    source_in_shm = source != Path("/dev/shm") and str(source).startswith(
        "/dev/shm/"
    )
    source_is_capture = (
        source.parent == assets_root
        and source.name.startswith(".p3-ge-external-capture-")
        and (source / ".cruise-ge-external-capture.json").is_file()
    )
    if source_is_capture:
        capture_marker = load_json(source / ".cruise-ge-external-capture.json")
        if (
            capture_marker.get("format") != "cruise-ge-external-capture-v1"
            or capture_marker.get("assets_root") != str(assets_root)
            or capture_marker.get("capture") != str(source)
        ):
            raise ValueError("source capture marker does not match its directory")
    if not source_in_shm and not source_is_capture:
        raise ValueError(
            "source must be /dev/shm scratch or a marker-owned XFS GE capture"
        )
    if not str(canonical).startswith(str(assets_root) + os.sep):
        raise ValueError("canonical bundle must be below assets root")
    if runtime_root != assets_root and not str(runtime_root).startswith(
        str(assets_root) + os.sep
    ):
        raise ValueError("runtime root must be assets root or one of its children")
    if not all(
        path.is_dir()
        for path in (source, canonical, assets_root, runtime_root)
    ):
        raise ValueError(
            "source, canonical, assets root, and runtime root must be directories"
        )
    if source_is_capture and source.stat().st_dev != runtime_root.stat().st_dev:
        raise ValueError("XFS capture and runtime root must share a filesystem")
    for bundle in reuse_bundles:
        if not str(bundle).startswith(str(assets_root) + os.sep) or not bundle.is_dir():
            raise ValueError(f"reusable bundle must be below assets root: {bundle}")

    canonical_by_content = canonical_content(canonical, reuse_bundles)
    records, generation_index, generation_count = source_records(
        source, args.generation_gap_seconds, args.generation
    )
    digest = identity(records)
    output = runtime_root / f"p3-ge-external-{digest[:16]}"
    hardlink_matches = sum(
        1
        for record in records
        if (int(record["bytes"]), str(record["sha256"])) in canonical_by_content
    )
    capture_hardlinks = len(records) - hardlink_matches if source_is_capture else 0
    unique_content = {
        (int(record["bytes"]), str(record["sha256"])) for record in records
    }
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "valid": True,
        "identity_sha256": digest,
        "source": str(source),
        "canonical": str(canonical),
        "reuse_bundles": [str(bundle) for bundle in reuse_bundles],
        "output": str(output),
        "logical_file_count": len(records),
        "logical_bytes": sum(int(record["bytes"]) for record in records),
        "unique_content_count": len(unique_content),
        "hardlink_matches": hardlink_matches,
        "capture_hardlinks": capture_hardlinks,
        "generation_selector": args.generation,
        "generation_index": generation_index,
        "generation_count": generation_count,
        "generation_mtime_min": min(float(record["mtime"]) for record in records),
        "generation_mtime_max": max(float(record["mtime"]) for record in records),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.apply:
        return 0
    ensure_runtime_traversal(assets_root)
    if output.exists():
        validate_promoted_bundle(
            output,
            source,
            records,
            digest,
            source_is_capture,
            canonical_by_content,
        )
        print(f"REUSE\t{output}")
        return 0

    stage = runtime_root / f".{output.name}.stage-{os.getpid()}"
    if stage.exists():
        raise ValueError(f"staging path already exists: {stage}")
    stage.mkdir(mode=0o700)
    try:
        hash_to_weight_file: dict[str, str] = {}
        for record in records:
            key = (int(record["bytes"]), str(record["sha256"]))
            destination = stage / str(record["name"])
            if key in canonical_by_content:
                os.link(canonical_by_content[key], destination)
            elif source_is_capture:
                os.link(Path(str(record["source"])), destination)
            else:
                shutil.copyfile(Path(str(record["source"])), destination)
            os.chmod(destination, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            hash_to_weight_file[str(record["hash"])] = str(output / destination.name)
        source_meta = load_json(source / "meta.json")
        promoted_meta = {
            "hash_to_weight_file": hash_to_weight_file,
            "hash_to_weight_offset": {
                key: value
                for key, value in dict(source_meta["hash_to_weight_offset"]).items()
                if key in hash_to_weight_file
            },
        }
        write_json(stage / "meta.json", promoted_meta)
        payload["valid"] = True
        payload["files"] = records
        write_json(stage / "dedup-manifest.json", payload)
        os.chmod(stage / "meta.json", stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(
            stage / "dedup-manifest.json", stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        )
        os.rename(stage, output)
        os.chmod(output, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    validate_promoted_bundle(
        output,
        source,
        records,
        digest,
        source_is_capture,
        canonical_by_content,
    )
    print(f"PROMOTE\t{output}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", required=True)
    result.add_argument("--canonical", required=True)
    result.add_argument("--assets-root", required=True)
    result.add_argument("--runtime-root", required=True)
    result.add_argument("--generation-gap-seconds", type=float, default=60.0)
    result.add_argument(
        "--generation", choices=("oldest", "latest", "all"), default="latest"
    )
    result.add_argument("--reuse-bundle", action="append", default=[])
    result.add_argument("--apply", action="store_true")
    return result


def main() -> int:
    try:
        return promote(parser().parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"P3 external-weight promotion error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
