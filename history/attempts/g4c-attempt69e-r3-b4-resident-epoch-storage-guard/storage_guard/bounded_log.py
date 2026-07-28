#!/usr/bin/env python3
"""Consume an unbounded byte stream while retaining a bounded diagnostic log."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--head-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--tail-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    if min(args.head_bytes, args.tail_bytes, args.chunk_bytes) < 0:
        parser.error("byte limits must be non-negative")
    if args.chunk_bytes == 0:
        parser.error("--chunk-bytes must be positive")
    return args


def update_tail(tail: bytearray, data: bytes, limit: int) -> None:
    if limit == 0:
        return
    if len(data) >= limit:
        tail[:] = data[-limit:]
        return
    excess = len(tail) + len(data) - limit
    if excess > 0:
        del tail[:excess]
    tail.extend(data)


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.metadata.exists():
        print("bounded_log refuses to overwrite an existing artifact", file=sys.stderr)
        return 2

    digest = hashlib.sha256()
    total_bytes = 0
    head_bytes = 0
    tail = bytearray()

    with args.output.open("xb") as output:
        while True:
            chunk = sys.stdin.buffer.read(args.chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            total_bytes += len(chunk)
            head_room = args.head_bytes - head_bytes
            if head_room > 0:
                prefix = chunk[:head_room]
                output.write(prefix)
                head_bytes += len(prefix)
                chunk = chunk[len(prefix) :]
            if chunk:
                update_tail(tail, chunk, args.tail_bytes)

        truncated = total_bytes > head_bytes + len(tail)
        if truncated:
            omitted = total_bytes - head_bytes - len(tail)
            marker = (
                "\n[storage-guard truncated "
                f"{omitted} bytes; full-stream-sha256={digest.hexdigest()}]\n"
            ).encode("ascii")
            output.write(marker)
        output.write(tail)
        output.flush()
        os.fsync(output.fileno())

    metadata = {
        "format": "bounded-log-v1",
        "total_bytes": total_bytes,
        "retained_stream_bytes": head_bytes + len(tail),
        "output_file_bytes": args.output.stat().st_size,
        "head_limit_bytes": args.head_bytes,
        "tail_limit_bytes": args.tail_bytes,
        "truncated": truncated,
        "omitted_bytes": total_bytes - head_bytes - len(tail),
        "full_stream_sha256": digest.hexdigest(),
    }
    with args.metadata.open("x", encoding="ascii", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
