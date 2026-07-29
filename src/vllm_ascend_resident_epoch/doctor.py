from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Literal

from .compatibility import (
    CompatibilityError,
    get_compatibility_profile,
    load_compatibility_manifest,
)
from .contract import ResidentEpochPlan, ResidentEpochRequest, ResidentEpochResult
from .runtime_config import CruiseRuntimeConfig, RuntimeConfigError
from .version import PYTHON_CONTRACT_VERSION


Status = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


@dataclass
class DoctorReport:
    mode: str
    profile: str | None
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def add(self, name: str, status: Status, detail: str) -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "profile": self.profile,
            "pass": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def _base_version(value: str) -> str:
    match = re.match(r"^(\d+\.\d+\.\d+)", value)
    return match.group(1) if match else value


def _module_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    value = getattr(module, "__version__", None)
    if not isinstance(value, str):
        raise RuntimeError(f"{module_name} does not expose __version__")
    return value


def _run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def run_source_smoke() -> DoctorReport:
    report = DoctorReport(mode="source-smoke", profile=None, checks=[])
    try:
        manifest = load_compatibility_manifest()
    except (CompatibilityError, OSError, json.JSONDecodeError) as exc:
        report.add("compatibility-manifest", "fail", str(exc))
    else:
        report.add(
            "compatibility-manifest",
            "pass",
            f"schema={manifest['schema_version']} profiles={len(manifest['profiles'])}",
        )

    if sys.version_info >= (3, 10):
        report.add("python", "pass", platform.python_version())
    else:
        report.add("python", "fail", f"requires Python >=3.10, found {platform.python_version()}")

    try:
        request = ResidentEpochRequest(
            req_id="smoke",
            row=0,
            generation=1,
            token_id=1,
            position=0,
            sequence_length=1,
            eos_token_id=2,
            scheduler_block_ids=(0,),
            device_block_ids=(0, 1),
        )
        plan = ResidentEpochPlan(
            version=PYTHON_CONTRACT_VERSION,
            graph_batch_size=4,
            max_steps=1,
            logical_capacity=8,
            requests=(request,),
            active_mask=(1, 0, 0, 0),
        )
        plan.validate()
        result = ResidentEpochResult(
            version=PYTHON_CONTRACT_VERSION,
            route="device",
            status=0,
            model_calls=1,
            computed_steps={"smoke": 1},
            row_generations=(1, 0, 0, 0),
        )
        result.validate_against(plan, {"smoke": [3]})
    except Exception as exc:
        report.add("contract-smoke", "fail", f"{type(exc).__name__}: {exc}")
    else:
        report.add("contract-smoke", "pass", "plan/result validation passed")
    return report


def _check_software(report: DoctorReport, profile: dict[str, Any]) -> None:
    software = profile["software"]
    observed_python = platform.python_version()
    report.add(
        "python-profile",
        "pass" if observed_python == software["python"] else "fail",
        f"expected={software['python']} observed={observed_python}",
    )
    modules = (
        ("torch", "torch", software["torch"], True),
        ("torch-npu", "torch_npu", software["torch_npu"], True),
        ("vllm", "vllm", software["vllm"]["version"], False),
        ("vllm-ascend", "vllm_ascend", software["vllm_ascend"]["version"], False),
    )
    for label, module, expected, compare_base in modules:
        try:
            observed = _module_version(module)
        except Exception as exc:
            report.add(label, "fail", f"import failed: {type(exc).__name__}: {exc}")
            continue
        left = _base_version(observed) if compare_base else observed
        right = _base_version(expected) if compare_base else expected
        report.add(
            label,
            "pass" if left == right else "fail",
            f"expected={expected} observed={observed}",
        )


def _find_cann_root(expected_version: str) -> Path | None:
    candidates: list[Path] = []
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        raw = os.getenv(name)
        if raw:
            candidates.append(Path(raw))
    candidates.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path(f"/usr/local/Ascend/cann-{expected_version}"),
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "opp" / "version.info").is_file():
            return resolved
    return None


def _check_cann(report: DoctorReport, profile: dict[str, Any]) -> None:
    expected = profile["software"]["cann"]
    root = _find_cann_root(expected)
    if root is None:
        report.add("cann", "fail", f"CANN {expected} installation was not found")
        return
    version_file = root / "opp" / "version.info"
    content = version_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^Version=(\S+)$", content, re.MULTILINE)
    observed = match.group(1) if match else "unknown"
    report.add(
        "cann",
        "pass" if observed == expected else "fail",
        f"expected={expected} observed={observed} root={root}",
    )


def _check_npu(report: DoctorReport, profile: dict[str, Any], device_id: int) -> None:
    executable = shutil.which("npu-smi")
    if executable is None:
        report.add("npu-smi", "fail", "npu-smi is not on PATH")
        return
    try:
        result = _run([executable, "info"], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.add("npu-smi", "fail", f"{type(exc).__name__}: {exc}")
        return
    if result.returncode != 0:
        report.add("npu-smi", "fail", f"exit={result.returncode}: {result.stderr.strip()}")
        return
    version_match = re.search(r"npu-smi\s+(\S+)", result.stdout)
    observed_version = version_match.group(1) if version_match else "unknown"
    expected_version = profile["hardware"]["npu_smi_version"]
    report.add(
        "npu-smi",
        "pass" if observed_version == expected_version else "fail",
        f"expected={expected_version} observed={observed_version}",
    )
    device_match = re.search(
        rf"^\|\s*{device_id}\s+(\S+)\s+\|\s+(\S+)",
        result.stdout,
        re.MULTILINE,
    )
    expected_accelerator = profile["hardware"]["accelerator"].split()[-1]
    if device_match is None:
        report.add("npu-device", "fail", f"device {device_id} was not listed")
        return
    product, health = device_match.groups()
    passed = product == expected_accelerator and health == "OK"
    report.add(
        "npu-device",
        "pass" if passed else "fail",
        f"device={device_id} product={product} health={health}",
    )


def run_npu_doctor(profile_id: str | None, device_id: int) -> DoctorReport:
    try:
        profile = get_compatibility_profile(profile_id)
    except CompatibilityError as exc:
        report = DoctorReport(mode="npu", profile=profile_id, checks=[])
        report.add("compatibility-profile", "fail", str(exc))
        return report
    report = DoctorReport(mode="npu", profile=profile["id"], checks=[])
    report.add("compatibility-profile", "pass", profile["id"])
    architecture = platform.machine()
    expected_architecture = profile["hardware"]["architecture"]
    report.add(
        "architecture",
        "pass" if architecture == expected_architecture else "fail",
        f"expected={expected_architecture} observed={architecture}",
    )
    _check_software(report, profile)
    _check_cann(report, profile)
    _check_npu(report, profile, device_id)
    report.add(
        "maturity",
        "warn",
        f"profile status={profile['status']}; this is not Stable v1.0",
    )
    return report


def run_runtime_doctor(config: CruiseRuntimeConfig, *, deep: bool) -> DoctorReport:
    report = run_npu_doctor(config.compatibility_profile, config.device_id)
    report.mode = "runtime"
    try:
        config.validate_paths(deep=deep)
    except RuntimeConfigError as exc:
        report.add("runtime-assets", "fail", str(exc))
    else:
        report.add(
            "runtime-assets",
            "pass",
            "all paths, contracts, counts, bytes, and configured hashes passed"
            + ("; external weights were deep-hashed" if deep else ""),
        )
    scratch_parent = _nearest_existing_parent(config.runtime.scratch_root)
    try:
        free = shutil.disk_usage(scratch_parent).free
    except OSError as exc:
        report.add("scratch-capacity", "fail", str(exc))
    else:
        required = config.runtime.minimum_scratch_free_bytes
        report.add(
            "scratch-capacity",
            "pass" if free >= required else "fail",
            f"required={required} free={free} filesystem={scratch_parent}",
        )
    return report


def render_report(report: DoctorReport, *, json_output: bool) -> str:
    if json_output:
        return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    lines = [f"Cruise {report.mode}: {'PASS' if report.passed else 'FAIL'}"]
    for check in report.checks:
        lines.append(f"{check.status.upper():4} {check.name}: {check.detail}")
    return "\n".join(lines) + "\n"
