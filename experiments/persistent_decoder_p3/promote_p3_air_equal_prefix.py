#!/usr/bin/env python3
"""Publish an XFS-staged P3 AIR with an equal-length path replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path


SCHEMA = 2
RUNTIME_AIR_NAME = "qwen_b4_p3_decoder_step.runtime.air"
RUNTIME_GRAPH_NAME = "dynamo.pbtxt"
SOURCE_RESULT_NAME = "source-export-result.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def replace_prefix(
    payload: bytes, old_prefix: bytes, new_prefix: bytes, expected_count: int
) -> bytes:
    if len(old_prefix) != len(new_prefix):
        raise ValueError("AIR prefix replacement must preserve byte length")
    count = payload.count(old_prefix)
    if count != expected_count:
        raise ValueError(
            f"AIR prefix occurrence mismatch: expected {expected_count}, observed {count}"
        )
    result = payload.replace(old_prefix, new_prefix)
    if old_prefix in result or len(result) != len(payload):
        raise ValueError("AIR prefix replacement did not preserve structure")
    return result


def file_constant_paths(graph: Path) -> list[Path]:
    text = graph.read_text(encoding="utf-8")
    return [
        Path(value)
        for value in re.findall(
            r'key: "file_path".*?s: \'s: "([^"]+)"\\n\'',
            text,
            re.DOTALL,
        )
    ]


def source_records(
    source_dir: Path, source_graph: Path, expected_count: int
) -> list[dict[str, object]]:
    paths = file_constant_paths(source_graph)
    if len(paths) != expected_count:
        raise ValueError(
            f"FileConstant count mismatch: expected {expected_count}, observed {len(paths)}"
        )
    if len(set(paths)) != len(paths):
        raise ValueError("FileConstant graph paths must be unique")
    records = []
    for path in sorted(paths, key=lambda item: item.name):
        resolved = path.resolve(strict=True)
        if resolved.parent != source_dir or not resolved.is_file():
            raise ValueError(f"FileConstant path escapes source directory: {path}")
        info = resolved.stat()
        records.append(
            {
                "name": resolved.name,
                "bytes": info.st_size,
                "sha256": sha256(resolved),
                "external": True,
                "source": str(resolved),
            }
        )
    return records


def manifest_identity(
    source_air_sha256: str,
    runtime_air_sha256: str,
    runtime_graph_sha256: str,
    records: list[dict[str, object]],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{source_air_sha256}\n{runtime_air_sha256}\n{runtime_graph_sha256}\n".encode(
            "ascii"
        )
    )
    for record in records:
        digest.update(
            f"{record['name']}\t{record['bytes']}\t{record['sha256']}\n".encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def validate_source_result(
    result: dict[str, object],
    source_air_sha256: str,
    source_graph_sha256: str,
    expected_count: int,
) -> None:
    required = {
        "pass": True,
        "air_sha256": source_air_sha256,
        "graph_sha256": source_graph_sha256,
        "external_file_count": expected_count,
        "eager_page_boundary_exact": True,
        "eager_stock_vllm_semantics_exact": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": result.get(key)}
        for key, expected in required.items()
        if result.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"source export result mismatch: {mismatches}")


def validate_published(
    output: Path, source_dir: Path | None = None
) -> dict[str, object]:
    manifest = load_json(output / "dedup-manifest.json")
    files = manifest.get("files")
    if not manifest.get("valid") or not isinstance(files, list):
        raise ValueError(f"invalid published manifest: {output}")
    for raw in files:
        if not isinstance(raw, dict):
            raise ValueError("invalid published file record")
        name = raw.get("name")
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            raise ValueError("invalid published file fields")
        path = output / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"published FileConstant mismatch: {path}")
        if source_dir is not None:
            source = source_dir / name
            if not source.is_file() or not os.path.samestat(source.stat(), path.stat()):
                raise ValueError(f"published FileConstant is not hard-linked: {path}")
        elif sha256(path) != digest:
            raise ValueError(f"published FileConstant digest mismatch: {path}")
    runtime_air = output / RUNTIME_AIR_NAME
    runtime_graph = output / RUNTIME_GRAPH_NAME
    if (
        sha256(runtime_air) != manifest.get("runtime_air_sha256")
        or sha256(runtime_graph) != manifest.get("runtime_graph_sha256")
    ):
        raise ValueError("published AIR or graph digest mismatch")
    return manifest


def promotion_summary(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
            "valid",
            "identity_sha256",
            "source_air_sha256",
            "runtime_air_sha256",
            "runtime_graph_sha256",
            "external_file_count",
            "logical_file_bytes",
            "hardlinked_file_count",
            "old_prefix",
            "new_prefix",
            "replacement_count",
        )
    }


def promote(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_dir).resolve(strict=True)
    source_air = Path(args.source_air).resolve(strict=True)
    source_graph = Path(args.source_graph).resolve(strict=True)
    source_result_path = Path(args.source_result).resolve(strict=True)
    assets_root = Path(args.assets_root).resolve(strict=True)
    output = Path(args.output).resolve(strict=False)
    expected_count = int(args.expected_external_count)
    if expected_count <= 0:
        raise ValueError("expected external count must be positive")
    if source_dir.parent != assets_root or source_air.parent != source_dir:
        raise ValueError("source must be a direct assets-root child containing the AIR")
    if source_graph.parent != source_dir or source_result_path.parent != source_dir:
        raise ValueError("source graph and export result must be direct source children")
    if output.parent != assets_root or output == source_dir or output.is_symlink():
        raise ValueError("output must be a distinct direct child of assets root")
    if source_dir.stat().st_dev != assets_root.stat().st_dev:
        raise ValueError("source staging and assets root must share a filesystem")

    original_air = source_air.read_bytes()
    original_graph = source_graph.read_bytes()
    source_air_sha256 = sha256_bytes(original_air)
    source_graph_sha256 = sha256_bytes(original_graph)
    expected_name = f"p3-air384-fia-{source_air_sha256[:16]}"
    if output.name != expected_name:
        raise ValueError(f"content-addressed output must be named {expected_name}")

    old_prefix = str(source_dir).encode("ascii")
    new_prefix = str(output).encode("ascii")
    runtime_air = replace_prefix(
        original_air, old_prefix, new_prefix, expected_count
    )
    runtime_graph = replace_prefix(
        original_graph, old_prefix, new_prefix, expected_count
    )
    source_result = load_json(source_result_path)
    validate_source_result(
        source_result, source_air_sha256, source_graph_sha256, expected_count
    )
    records = source_records(source_dir, source_graph, expected_count)
    runtime_air_sha256 = sha256_bytes(runtime_air)
    runtime_graph_sha256 = sha256_bytes(runtime_graph)
    identity = manifest_identity(
        source_air_sha256, runtime_air_sha256, runtime_graph_sha256, records
    )
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "valid": True,
        "identity_sha256": identity,
        "source_dir": str(source_dir),
        "source_air": str(source_air),
        "source_air_bytes": len(original_air),
        "source_air_sha256": source_air_sha256,
        "source_graph": str(source_graph),
        "source_graph_sha256": source_graph_sha256,
        "source_export_result_sha256": sha256(source_result_path),
        "runtime_air": str(output / RUNTIME_AIR_NAME),
        "runtime_air_bytes": len(runtime_air),
        "runtime_air_sha256": runtime_air_sha256,
        "runtime_graph": str(output / RUNTIME_GRAPH_NAME),
        "runtime_graph_bytes": len(runtime_graph),
        "runtime_graph_sha256": runtime_graph_sha256,
        "old_prefix": str(source_dir),
        "new_prefix": str(output),
        "prefix_bytes": len(old_prefix),
        "replacement_count": expected_count,
        "external_file_count": len(records),
        "hardlinked_file_count": len(records),
        "logical_file_bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }
    print(json.dumps(promotion_summary(manifest), indent=2, sort_keys=True))
    if not args.apply:
        return 0

    if output.exists():
        existing = validate_published(output, source_dir)
        if existing.get("identity_sha256") != identity:
            raise ValueError(f"existing output has a different identity: {output}")
        print(f"REUSE\t{output}")
        return 0

    stage = assets_root / f".{output.name}.stage-{os.getpid()}"
    if stage.exists() or stage.is_symlink():
        raise ValueError(f"staging path already exists: {stage}")
    stage.mkdir(mode=0o700)
    try:
        for record in records:
            source = source_dir / str(record["name"])
            destination = stage / source.name
            os.link(source, destination)
            os.chmod(destination, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        (stage / RUNTIME_AIR_NAME).write_bytes(runtime_air)
        (stage / RUNTIME_GRAPH_NAME).write_bytes(runtime_graph)
        shutil.copyfile(source_result_path, stage / SOURCE_RESULT_NAME)
        write_json(stage / "dedup-manifest.json", manifest)
        for name in (
            RUNTIME_AIR_NAME,
            RUNTIME_GRAPH_NAME,
            SOURCE_RESULT_NAME,
            "dedup-manifest.json",
        ):
            path = stage / name
            os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(stage, output)
        os.chmod(
            output,
            stat.S_IRUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH,
        )
        parent_fd = os.open(assets_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    validate_published(output, source_dir)
    print(f"PROMOTE\t{output}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", required=True)
    result.add_argument("--source-air", required=True)
    result.add_argument("--source-graph", required=True)
    result.add_argument("--source-result", required=True)
    result.add_argument("--assets-root", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--expected-external-count", type=int, default=202)
    result.add_argument("--apply", action="store_true")
    return result


def main() -> int:
    try:
        return promote(parser().parse_args())
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        print(f"P3 AIR promotion error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
