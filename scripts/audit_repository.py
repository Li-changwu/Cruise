#!/usr/bin/env python3
"""Reject generated artifacts, oversized files, and likely secrets."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024

FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "archive",
    "results",
    "scratch",
    "runtime",
    "weights",
    "models",
    "profiling",
    "kernel_meta",
    "opp_kernel_cache",
    "ge_cache",
}
FORBIDDEN_SUFFIXES = {
    ".air",
    ".om",
    ".o",
    ".so",
    ".a",
    ".bin",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
    ".onnx",
    ".log",
    ".trace",
    ".prof",
    ".sqlite",
    ".sock",
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
}

SECRET_PATTERNS = {
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "OpenAI-style key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "original SSH alias": re.compile(rb"\bvllm-hust-lcw-21rc\b"),
}


def repository_files() -> list[Path]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        ROOT / raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw
    ]


def main() -> int:
    failures: list[str] = []
    files = repository_files()
    total_bytes = 0

    for path in files:
        relative = path.relative_to(ROOT)
        normalized_parts = {part.lower() for part in relative.parts}
        if normalized_parts & FORBIDDEN_PARTS:
            failures.append(f"forbidden directory: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact type: {relative}")
        if not path.is_file():
            continue

        size = path.stat().st_size
        total_bytes += size
        if size > MAX_FILE_BYTES:
            failures.append(
                f"oversized file: {relative} ({size / 1024 / 1024:.2f} MiB)"
            )

        if size <= MAX_FILE_BYTES:
            content = path.read_bytes()
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(content):
                    failures.append(f"{label}: {relative}")

    if total_bytes > MAX_TOTAL_BYTES:
        failures.append(
            f"repository payload is {total_bytes / 1024 / 1024:.2f} MiB; "
            f"limit is {MAX_TOTAL_BYTES / 1024 / 1024:.0f} MiB"
        )

    if failures:
        print("Repository audit failed:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"Repository audit passed: {len(files)} files, "
        f"{total_bytes / 1024 / 1024:.2f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
