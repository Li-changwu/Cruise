#!/usr/bin/env python3
"""Audit the public FIA package and its source-preserving kernel hierarchy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "afe72144f9f2ac8441929035795db88a111b30c5"
FAMILIES = (
    "fused_infer_attention_score",
    "incre_flash_attention",
    "prompt_flash_attention",
    "common",
)
ENTRY_INCLUDES = (
    "../../incre_flash_attention/op_kernel/incre_flash_attention_arch32.h",
    "../../prompt_flash_attention/op_kernel/prompt_flash_attention_arch32.h",
)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        top = relative.parts[0]
        if top.startswith("arch") and top != "arch32":
            continue
        selected.append(relative)
    return selected


def tree_digest(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(root / relative)))
    return digest.hexdigest()


def git_value(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    install_root = args.install_root.resolve(strict=True)
    patch = args.patch.resolve(strict=True)
    vendors = sorted((install_root / "vendors").glob("*_transformer"))
    impl_roots = (
        sorted(vendors[0].glob("op_impl/ai_core/tbe/*_transformer_impl"))
        if len(vendors) == 1
        else []
    )
    source_commit = git_value(source, "rev-parse", "HEAD")
    source_status = git_value(source, "status", "--porcelain=v1", "--untracked-files=no")
    source_remote = git_value(source, "remote", "get-url", "origin")

    checks: dict[str, bool] = {
        "public_source_commit": source_commit == args.expected_commit,
        "tracked_source_clean": not source_status,
        "single_generated_vendor": len(vendors) == 1,
        "single_impl_root": len(impl_roots) == 1,
        "isolated_install_root": str(install_root).startswith("/dev/shm/"),
        "packaging_patch_present": patch.is_file(),
    }
    family_records: dict[str, object] = {}
    cross_family_targets: list[dict[str, object]] = []

    if len(impl_roots) == 1:
        impl_root = impl_roots[0]
        for family in FAMILIES:
            source_root = source / "attention" / family / "op_kernel"
            installed_root = impl_root / "ascendc" / family / "op_kernel"
            expected = source_files(source_root) if source_root.is_dir() else []
            missing = [
                relative.as_posix()
                for relative in expected
                if not (installed_root / relative).is_file()
            ]
            mismatched = [
                relative.as_posix()
                for relative in expected
                if (installed_root / relative).is_file()
                and sha256(source_root / relative)
                != sha256(installed_root / relative)
            ]
            family_pass = bool(expected) and not missing and not mismatched
            checks[f"{family}_hierarchy_exact"] = family_pass
            family_records[family] = {
                "source_file_count": len(expected),
                "missing": missing,
                "mismatched": mismatched,
                "source_tree_sha256": tree_digest(source_root, expected),
                "installed_tree_sha256": (
                    tree_digest(installed_root, expected) if not missing else None
                ),
            }

        entry = (
            impl_root
            / "ascendc"
            / "fused_infer_attention_score"
            / "op_kernel"
            / "fused_infer_attention_score.cpp"
        )
        found_includes = []
        if entry.is_file():
            for line in entry.read_text(encoding="utf-8").splitlines():
                match = INCLUDE_RE.match(line)
                if match:
                    found_includes.append(match.group(1))
        for include in ENTRY_INCLUDES:
            target = (entry.parent / include).resolve()
            resolved = target.is_file()
            cross_family_targets.append(
                {"include": include, "target": str(target), "resolved": resolved}
            )
            checks[f"entry_include_{Path(include).name}"] = (
                include in found_includes and resolved
            )
        checks["dynamic_compile_entry_present"] = entry.is_file()

        config = (
            vendors[0]
            / "op_impl"
            / "ai_core"
            / "tbe"
            / "config"
            / "ascend910b"
            / "aic-ascend910b-ops-info.json"
        )
        dynamic = impl_root / "dynamic" / "fused_infer_attention_score.py"
        checks["generated_registration_present"] = config.is_file()
        checks["generated_dynamic_compiler_present"] = dynamic.is_file()

    result = {
        "schema_version": 1,
        "pass": all(checks.values()),
        "source": {
            "commit": source_commit,
            "remote": source_remote,
            "tracked_clean": not source_status,
        },
        "package": {
            "install_root": str(install_root),
            "vendor": str(vendors[0]) if len(vendors) == 1 else None,
            "patch_sha256": sha256(patch) if patch.is_file() else None,
        },
        "checks": checks,
        "families": family_records,
        "cross_family_targets": cross_family_targets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
