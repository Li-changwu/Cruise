from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Sequence

from .compatibility import CompatibilityError, load_compatibility_manifest
from .doctor import (
    render_report,
    run_npu_doctor,
    run_runtime_doctor,
    run_source_smoke,
)
from .runtime_config import CruiseRuntimeConfig, RuntimeConfigError, load_runtime_config
from .version import PACKAGE_VERSION, PRODUCT_MATURITY


_SCRATCH_MARKER = ".cruise-scratch-root-v1"
_RUN_MARKER = ".cruise-run-v1"
_MARKER_CONTENT = "Cruise managed scratch v1\n"


def _print_config(config: CruiseRuntimeConfig, *, paths_checked: bool) -> None:
    print(
        json.dumps(
            {
                "schema_version": 1,
                "pass": True,
                "configuration": str(config.source),
                "profile": config.compatibility_profile,
                "device_id": config.device_id,
                "max_steps": config.runtime.max_steps,
                "logical_capacity": config.runtime.logical_capacity,
                "paths_checked": paths_checked,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _ensure_scratch_root(config: CruiseRuntimeConfig) -> Path:
    root = config.runtime.scratch_root.resolve()
    if os.name != "posix":
        raise RuntimeConfigError("cruise run is supported only on Linux")
    try:
        root.relative_to("/dev/shm")
    except ValueError as exc:
        raise RuntimeConfigError(f"scratch_root must be below /dev/shm: {root}") from exc
    if root == Path("/dev/shm"):
        raise RuntimeConfigError("scratch_root must not be /dev/shm itself")

    marker = root / _SCRATCH_MARKER
    if root.exists():
        if not root.is_dir():
            raise RuntimeConfigError(f"scratch_root is not a directory: {root}")
        if marker.exists():
            if marker.read_text(encoding="ascii") != _MARKER_CONTENT:
                raise RuntimeConfigError(f"scratch marker is invalid: {marker}")
        else:
            entries = list(root.iterdir())
            if entries:
                raise RuntimeConfigError(
                    f"refusing to adopt non-empty unmarked scratch_root: {root}"
                )
            marker.write_text(_MARKER_CONTENT, encoding="ascii")
    else:
        root.mkdir(parents=True, mode=0o700)
        marker.write_text(_MARKER_CONTENT, encoding="ascii")
    free = shutil.disk_usage(root).free
    required = config.runtime.minimum_scratch_free_bytes
    if free < required:
        raise RuntimeConfigError(
            f"scratch filesystem has {free} free bytes; {required} required"
        )
    return root


def _create_run_directory(root: Path) -> Path:
    path = Path(tempfile.mkdtemp(prefix="run-", dir=root)).resolve()
    (path / _RUN_MARKER).write_text(_MARKER_CONTENT, encoding="ascii")
    return path


def _cleanup_run_directory(root: Path, run_directory: Path) -> None:
    resolved_root = root.resolve()
    resolved_run = run_directory.resolve()
    if resolved_run.parent != resolved_root:
        raise RuntimeConfigError(f"refusing to clean unexpected run directory: {resolved_run}")
    marker = resolved_run / _RUN_MARKER
    if not marker.is_file() or marker.read_text(encoding="ascii") != _MARKER_CONTENT:
        raise RuntimeConfigError(f"refusing to clean unmarked run directory: {resolved_run}")
    shutil.rmtree(resolved_run)


def _remove_empty_scratch_root(config: CruiseRuntimeConfig) -> None:
    root = config.runtime.scratch_root.resolve()
    marker = root / _SCRATCH_MARKER
    if not root.exists():
        return
    if not root.is_dir() or not marker.is_file():
        raise RuntimeConfigError(f"refusing to remove unmarked scratch_root: {root}")
    if marker.read_text(encoding="ascii") != _MARKER_CONTENT:
        raise RuntimeConfigError(f"scratch marker is invalid: {marker}")
    entries = [path for path in root.iterdir() if path != marker]
    if entries:
        rendered = ", ".join(path.name for path in sorted(entries))
        raise RuntimeConfigError(f"scratch_root is not empty: {rendered}")
    marker.unlink()
    root.rmdir()


def _run_supervised(config: CruiseRuntimeConfig, command: list[str], *, deep: bool) -> int:
    if not command:
        raise RuntimeConfigError("cruise run requires a command after --")
    config.validate_paths(deep=deep)
    root = _ensure_scratch_root(config)
    run_directory = _create_run_directory(root)
    environment = config.environment(run_directory, os.environ)
    shell_script = 'set -euo pipefail; source "$1"; shift; exec "$@"'
    arguments = [
        "/bin/bash",
        "-c",
        shell_script,
        "cruise",
        str(config.cann_set_env),
        *command,
    ]
    process: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, signal.Handlers] = {}

    def forward(signum: int, _frame: object) -> None:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)

    try:
        process = subprocess.Popen(arguments, env=environment, start_new_session=True)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward)
        return process.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=60)
        _cleanup_run_directory(root, run_directory)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cruise")
    parser.add_argument("--version", action="version", version=f"Cruise {PACKAGE_VERSION}")
    commands = parser.add_subparsers(dest="command", required=True)

    version_parser = commands.add_parser("version", help="show product and contract versions")
    version_parser.add_argument("--json", action="store_true", dest="json_output")

    doctor = commands.add_parser("doctor", help="validate source, NPU, or runtime readiness")
    doctor.add_argument("--mode", choices=("source", "npu", "runtime"), default="source")
    doctor.add_argument("--profile")
    doctor.add_argument("--device", type=int, default=0)
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--deep", action="store_true")
    doctor.add_argument("--json", action="store_true", dest="json_output")

    smoke = commands.add_parser("smoke", help="run the bounded no-NPU contract smoke")
    smoke.add_argument("--json", action="store_true", dest="json_output")

    config_parser = commands.add_parser("config", help="runtime configuration operations")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="validate a runtime configuration")
    validate.add_argument("path", type=Path)
    validate.add_argument("--check-paths", action="store_true")
    validate.add_argument("--deep", action="store_true")

    run = commands.add_parser("run", help="validate and run a command in managed scratch")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--deep", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("child_command", nargs=argparse.REMAINDER)

    cleanup = commands.add_parser("cleanup", help="remove an empty managed scratch root")
    cleanup.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            manifest = load_compatibility_manifest()
            value = {
                "product": "Cruise",
                "version": PACKAGE_VERSION,
                "maturity": PRODUCT_MATURITY,
                "contracts": manifest["contracts"],
            }
            if args.json_output:
                print(json.dumps(value, indent=2, sort_keys=True))
            else:
                print(f"Cruise {PACKAGE_VERSION} ({PRODUCT_MATURITY})")
                for name, version in value["contracts"].items():
                    print(f"{name}={version}")
            return 0

        if args.command == "smoke":
            report = run_source_smoke()
            print(render_report(report, json_output=args.json_output), end="")
            return 0 if report.passed else 1

        if args.command == "doctor":
            if args.mode == "source":
                if args.config is not None:
                    parser.error("--config is valid only with --mode runtime")
                report = run_source_smoke()
                report.mode = "source"
            elif args.mode == "npu":
                if args.config is not None:
                    parser.error("--config is valid only with --mode runtime")
                report = run_npu_doctor(args.profile, args.device)
            else:
                if args.config is None:
                    parser.error("--mode runtime requires --config")
                config = load_runtime_config(args.config)
                report = run_runtime_doctor(config, deep=args.deep)
            print(render_report(report, json_output=args.json_output), end="")
            return 0 if report.passed else 1

        if args.command == "config":
            config = load_runtime_config(args.path)
            if args.check_paths or args.deep:
                config.validate_paths(deep=args.deep)
            _print_config(config, paths_checked=args.check_paths or args.deep)
            return 0

        if args.command == "run":
            config = load_runtime_config(args.config)
            command = list(args.child_command)
            if command and command[0] == "--":
                command = command[1:]
            if not command:
                parser.error("cruise run requires a command after --")
            if args.dry_run:
                config.validate_paths(deep=args.deep)
                print(
                    json.dumps(
                        {
                            "pass": True,
                            "profile": config.compatibility_profile,
                            "device_id": config.device_id,
                            "command": command,
                            "deep_integrity": args.deep,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            return _run_supervised(config, command, deep=args.deep)

        if args.command == "cleanup":
            config = load_runtime_config(args.config)
            _remove_empty_scratch_root(config)
            print(f"Cruise scratch root is absent: {config.runtime.scratch_root}")
            return 0
    except (RuntimeConfigError, CompatibilityError, OSError, json.JSONDecodeError) as exc:
        print(f"cruise: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
