#!/usr/bin/env python3
"""Create and remove a bounded mutable GE view over canonical weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path


MARKER = ".cruise-ge-external-view.json"
CAPTURE_MARKER = ".cruise-ge-external-capture.json"


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def exact_view(run_root: Path, raw_view: str) -> Path:
    expected = run_root / ".ge-external-view"
    view = Path(raw_view).absolute()
    if view != expected:
        raise ValueError(f"GE view must be the exact run child: {expected}")
    return view


def exact_capture(assets_root: Path, raw_capture: str) -> Path:
    capture = Path(raw_capture).absolute()
    if (
        capture.parent != assets_root
        or not capture.name.startswith(".p3-ge-external-capture-")
        or capture.is_symlink()
    ):
        raise ValueError(
            "GE capture must be a non-symlink direct assets-root child named "
            ".p3-ge-external-capture-*"
        )
    return capture


def validate_source(source: Path) -> tuple[dict[str, object], dict[str, object]]:
    meta_path = source / "meta.json"
    manifest = load_json(source / "dedup-manifest.json")
    meta = load_json(meta_path)
    mapping = meta.get("hash_to_weight_file")
    if not manifest.get("valid"):
        raise ValueError("canonical external-weight manifest is not valid")
    if not isinstance(mapping, dict) or len(mapping) != manifest.get(
        "logical_file_count"
    ):
        raise ValueError("canonical external-weight meta and manifest counts disagree")
    for raw_path in mapping.values():
        if not isinstance(raw_path, str):
            raise ValueError("canonical external-weight path must be a string")
        path = Path(raw_path).resolve(strict=True)
        if path.parent != source or not path.is_file():
            raise ValueError(f"canonical external-weight path escapes bundle: {path}")
    return meta, manifest


def create(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve(strict=True)
    run_root = Path(args.run_root).resolve(strict=True)
    view = exact_view(run_root, args.view)
    if view.exists() or view.is_symlink():
        raise ValueError(f"GE view already exists: {view}")
    meta, manifest = validate_source(source)
    view.mkdir(mode=0o700)
    try:
        mapping = dict(meta["hash_to_weight_file"])
        view_mapping: dict[str, str] = {}
        for weight_hash, raw_path in mapping.items():
            source_weight = Path(str(raw_path)).resolve(strict=True)
            link = view / source_weight.name
            os.symlink(source_weight, link)
            view_mapping[str(weight_hash)] = str(link)
        meta["hash_to_weight_file"] = view_mapping
        write_json(view / "meta.json", meta)
        os.chmod(view / "meta.json", stat.S_IRUSR | stat.S_IWUSR)
        write_json(
            view / MARKER,
            {
                "format": "cruise-ge-external-view-v1",
                "run_root": str(run_root),
                "source": str(source),
                "source_identity_sha256": manifest.get("identity_sha256"),
                "view": str(view),
            },
        )
    except BaseException:
        shutil.rmtree(view)
        raise
    print(view)
    return 0


def capture_create(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve(strict=True)
    assets_root = Path(args.assets_root).resolve(strict=True)
    capture = exact_capture(assets_root, args.capture)
    if capture.exists():
        raise ValueError(f"GE capture already exists: {capture}")
    if source.stat().st_dev != assets_root.stat().st_dev:
        raise ValueError("GE source and capture root must share a filesystem")
    meta, manifest = validate_source(source)
    capture.mkdir(mode=0o700)
    try:
        mapping = dict(meta["hash_to_weight_file"])
        capture_mapping: dict[str, str] = {}
        for weight_hash, raw_path in mapping.items():
            source_weight = Path(str(raw_path)).resolve(strict=True)
            destination = capture / source_weight.name
            os.link(source_weight, destination)
            capture_mapping[str(weight_hash)] = str(destination)
        meta["hash_to_weight_file"] = capture_mapping
        write_json(capture / "meta.json", meta)
        os.chmod(capture / "meta.json", stat.S_IRUSR | stat.S_IWUSR)
        write_json(
            capture / CAPTURE_MARKER,
            {
                "format": "cruise-ge-external-capture-v1",
                "assets_root": str(assets_root),
                "source": str(source),
                "source_identity_sha256": manifest.get("identity_sha256"),
                "capture": str(capture),
                "seed_file_count": len(capture_mapping),
            },
        )
        os.chmod(capture / CAPTURE_MARKER, stat.S_IRUSR | stat.S_IWUSR)
    except BaseException:
        shutil.rmtree(capture)
        raise
    print(capture)
    return 0


def allocated_bytes(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in (root, *root.rglob("*")):
        info = path.lstat()
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += info.st_blocks * 512
    return total


def exclusive_allocated_bytes(root: Path) -> int:
    total = 0
    for path in (root, *root.rglob("*")):
        info = path.lstat()
        if info.st_nlink == 1:
            total += info.st_blocks * 512
    return total


def cleanup(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve(strict=True)
    view = exact_view(run_root, args.view)
    receipt = Path(args.receipt).absolute()
    evidence = run_root / "evidence"
    if receipt.parent != evidence:
        raise ValueError(f"GE view receipt must be below {evidence}")
    marker = load_json(view / MARKER)
    if (
        marker.get("format") != "cruise-ge-external-view-v1"
        or marker.get("run_root") != str(run_root)
        or marker.get("view") != str(view)
    ):
        raise ValueError("GE view marker does not match cleanup target")
    meta = view / "meta.json"
    meta_bytes = meta.stat().st_size if meta.is_file() else 0
    meta_sha256 = (
        hashlib.sha256(meta.read_bytes()).hexdigest() if meta.is_file() else None
    )
    payload = {
        "format": "cruise-ge-external-view-cleanup-v1",
        "view": str(view),
        "source": marker.get("source"),
        "allocated_bytes_before_cleanup": allocated_bytes(view),
        "file_count_before_cleanup": sum(1 for path in view.rglob("*") if path.is_file()),
        "meta_bytes_before_cleanup": meta_bytes,
        "meta_sha256_before_cleanup": meta_sha256,
    }
    write_json(receipt, payload)
    shutil.rmtree(view)
    print(receipt)
    return 0


def capture_cleanup(args: argparse.Namespace) -> int:
    assets_root = Path(args.assets_root).resolve(strict=True)
    run_root = Path(args.run_root).resolve(strict=True)
    capture = exact_capture(assets_root, args.capture)
    receipt = Path(args.receipt).absolute()
    evidence = run_root / "evidence"
    if receipt.parent != evidence:
        raise ValueError(f"GE capture receipt must be below {evidence}")
    marker = load_json(capture / CAPTURE_MARKER)
    if (
        marker.get("format") != "cruise-ge-external-capture-v1"
        or marker.get("assets_root") != str(assets_root)
        or marker.get("capture") != str(capture)
    ):
        raise ValueError("GE capture marker does not match cleanup target")
    meta = capture / "meta.json"
    payload = {
        "format": "cruise-ge-external-capture-cleanup-v1",
        "capture": str(capture),
        "source": marker.get("source"),
        "allocated_bytes_before_cleanup": allocated_bytes(capture),
        "exclusive_allocated_bytes_before_cleanup": exclusive_allocated_bytes(
            capture
        ),
        "file_count_before_cleanup": sum(
            1 for path in capture.rglob("*") if path.is_file()
        ),
        "meta_bytes_before_cleanup": meta.stat().st_size if meta.is_file() else 0,
        "meta_sha256_before_cleanup": (
            hashlib.sha256(meta.read_bytes()).hexdigest() if meta.is_file() else None
        ),
    }
    write_json(receipt, payload)
    shutil.rmtree(capture)
    print(receipt)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--source", required=True)
    create_parser.add_argument("--run-root", required=True)
    create_parser.add_argument("--view", required=True)
    create_parser.set_defaults(handler=create)
    capture_create_parser = subparsers.add_parser("capture-create")
    capture_create_parser.add_argument("--source", required=True)
    capture_create_parser.add_argument("--assets-root", required=True)
    capture_create_parser.add_argument("--capture", required=True)
    capture_create_parser.set_defaults(handler=capture_create)
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--run-root", required=True)
    cleanup_parser.add_argument("--view", required=True)
    cleanup_parser.add_argument("--receipt", required=True)
    cleanup_parser.set_defaults(handler=cleanup)
    capture_cleanup_parser = subparsers.add_parser("capture-cleanup")
    capture_cleanup_parser.add_argument("--assets-root", required=True)
    capture_cleanup_parser.add_argument("--run-root", required=True)
    capture_cleanup_parser.add_argument("--capture", required=True)
    capture_cleanup_parser.add_argument("--receipt", required=True)
    capture_cleanup_parser.set_defaults(handler=capture_cleanup)
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        return args.handler(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"GE external view error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
