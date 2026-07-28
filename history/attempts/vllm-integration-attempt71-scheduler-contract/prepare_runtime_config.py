#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--controller-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.controller_workspace.resolve(strict=True)
    if not str(workspace).startswith("/dev/shm/"):
        raise RuntimeError("controller workspace must be under /dev/shm")
    for name in ("CMakeLists.txt", "g4c_b4_resident_epoch.cpp"):
        if not (workspace / name).is_file():
            raise RuntimeError(f"missing controller source: {workspace / name}")

    config = json.loads(args.template.read_text(encoding="utf-8"))
    config["workspace"] = str(workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
