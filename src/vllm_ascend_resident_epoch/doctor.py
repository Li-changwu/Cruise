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
    get_capability_requirements,
    get_compatibility_profile,
    list_compatibility_profiles,
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
    code: str | None = None
    expected: str | None = None
    observed: str | None = None
    remediation: str | None = None


@dataclass
class DoctorReport:
    mode: str
    profile: str | None
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def add(
        self,
        name: str,
        status: Status,
        detail: str,
        *,
        code: str | None = None,
        expected: str | None = None,
        observed: str | None = None,
        remediation: str | None = None,
    ) -> None:
        self.checks.append(
            Check(
                name=name,
                status=status,
                detail=detail,
                code=code,
                expected=expected,
                observed=observed,
                remediation=remediation,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "mode": self.mode,
            "profile": self.profile,
            "pass": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def _base_version(value: str) -> str:
    match = re.match(r"^(\d+\.\d+\.\d+)", value)
    return match.group(1) if match else value


def _requirement_detail(expected: object, observed: object, remediation: str) -> str:
    return (
        f"expected={expected}; observed={observed}; "
        f"remediation={remediation}"
    )


def _add_requirement(
    report: DoctorReport,
    name: str,
    passed: bool,
    *,
    code: str,
    expected: object,
    observed: object,
    remediation: str,
) -> None:
    report.add(
        name,
        "pass" if passed else "fail",
        _requirement_detail(expected, observed, remediation),
        code=None if passed else code,
        expected=str(expected),
        observed=str(observed),
        remediation=remediation,
    )


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
    _add_requirement(
        report,
        "python-profile",
        observed_python == software["python"],
        code="incompatible-python-version",
        expected=software["python"],
        observed=observed_python,
        remediation=(
            f"activate an environment with Python {software['python']} or select a "
            "profile validated for this interpreter"
        ),
    )
    modules = (
        ("torch", "torch", software["torch"], True),
        ("torch-npu", "torch_npu", software["torch_npu"], False),
        ("vllm", "vllm", software["vllm"]["version"], False),
        ("vllm-ascend", "vllm_ascend", software["vllm_ascend"]["version"], False),
    )
    for label, module, expected, compare_base in modules:
        try:
            observed = _module_version(module)
        except Exception as exc:
            observed = f"import failed: {type(exc).__name__}: {exc}"
            remediation = (
                f"install {label} from the selected profile into the active Python "
                "environment"
            )
            report.add(
                label,
                "fail",
                _requirement_detail(expected, observed, remediation),
                code="missing-python-dependency",
                expected=expected,
                observed=observed,
                remediation=remediation,
            )
            continue
        left = _base_version(observed) if compare_base else observed
        right = _base_version(expected) if compare_base else expected
        _add_requirement(
            report,
            label,
            left == right,
            code="incompatible-package-version",
            expected=expected,
            observed=observed,
            remediation=(
                f"install the {label} build declared by profile {profile['id']} or "
                "qualify this package combination as a new profile"
            ),
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


def _check_cann(report: DoctorReport, profile: dict[str, Any]) -> Path | None:
    expected = profile["software"]["cann"]
    root = _find_cann_root(expected)
    if root is None:
        remediation = (
            f"install CANN {expected} and source its set_env.sh before running doctor"
        )
        report.add(
            "cann",
            "fail",
            _requirement_detail(expected, "not found", remediation),
            code="missing-cann-runtime",
            expected=expected,
            observed="not found",
            remediation=remediation,
        )
        return None
    version_file = root / "opp" / "version.info"
    content = version_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^Version=(\S+)$", content, re.MULTILINE)
    observed = match.group(1) if match else "unknown"
    _add_requirement(
        report,
        "cann",
        observed == expected,
        code="incompatible-cann-version",
        expected=expected,
        observed=f"{observed} at {root}",
        remediation=(
            f"source CANN {expected} or select a profile validated for CANN {observed}"
        ),
    )
    return root


def _find_cann_component(root: Path, component: dict[str, Any]) -> Path | None:
    basename = component["name"]
    executable = shutil.which(basename)
    if executable:
        return Path(executable).resolve()
    for relative in component["relative_paths"]:
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def _check_cann_capabilities(
    report: DoctorReport,
    requirements: dict[str, Any],
    cann_root: Path | None,
) -> None:
    for module_name in requirements["required_python_modules"]:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            installed_site = (
                cann_root / "python" / "site-packages"
                if cann_root is not None
                else None
            )
            installed_module = (
                installed_site / module_name
                if installed_site is not None
                else None
            )
            if installed_module is not None and installed_module.is_dir():
                observed = (
                    f"installed at {installed_module} but import failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                remediation = (
                    f"export PYTHONPATH={installed_site}:$PYTHONPATH in the active "
                    f"Python {platform.python_version()} environment, then rerun doctor"
                )
                code = "inactive-device-control-runtime"
            else:
                observed = f"import failed: {type(exc).__name__}: {exc}"
                remediation = (
                    "install the CANN DataFlow Python package in the active environment "
                    "and source the toolkit set_env.sh"
                )
                code = "missing-device-control-runtime"
            report.add(
                f"python-module-{module_name}",
                "fail",
                _requirement_detail("importable", observed, remediation),
                code=code,
                expected="importable",
                observed=observed,
                remediation=remediation,
            )
        else:
            module_path = getattr(module, "__file__", "built-in")
            _add_requirement(
                report,
                f"python-module-{module_name}",
                True,
                code="missing-device-control-runtime",
                expected="importable",
                observed=module_path,
                remediation="none",
            )

    for symbol in requirements["required_python_symbols"]:
        expected = "one of " + ", ".join(
            f"{item['module']}.{item['attribute']}"
            for item in symbol["alternatives"]
        )
        matched: str | None = None
        failures: list[str] = []
        for alternative in symbol["alternatives"]:
            qualified = f"{alternative['module']}.{alternative['attribute']}"
            try:
                value: object = importlib.import_module(alternative["module"])
                for name in alternative["attribute"].split("."):
                    value = getattr(value, name)
            except Exception as exc:
                failures.append(f"{qualified}: {type(exc).__name__}: {exc}")
            else:
                matched = qualified
                break
        if matched is None:
            observed = "unavailable: " + "; ".join(failures)
            _add_requirement(
                report,
                f"python-symbol-{symbol['id']}",
                False,
                code="missing-python-runtime-capability",
                expected=expected,
                observed=observed,
                remediation=symbol["remediation"],
            )
        else:
            _add_requirement(
                report,
                f"python-symbol-{symbol['id']}",
                True,
                code="missing-python-runtime-capability",
                expected=expected,
                observed=f"available as {matched}",
                remediation="none",
            )

    for component in requirements["required_cann_files"]:
        component_id = component["id"]
        basename = component["name"]
        found = (
            _find_cann_component(cann_root, component)
            if cann_root is not None
            else None
        )
        remediation = (
            f"install the CANN component that provides {basename} and source "
            "the matching toolkit environment"
        )
        _add_requirement(
            report,
            component_id,
            found is not None,
            code="missing-cann-component",
            expected=f"CANN file {basename}",
            observed=str(found) if found is not None else "not found",
            remediation=remediation,
        )


def _check_shared_memory(
    report: DoctorReport, requirements: dict[str, Any]
) -> None:
    path = Path("/dev/shm")
    required = requirements["minimum_shared_memory_free_bytes"]
    if os.name != "posix":
        _add_requirement(
            report,
            "shared-memory",
            False,
            code="unsupported-shared-memory-filesystem",
            expected="writable /dev/shm",
            observed=f"os.name={os.name}",
            remediation="run Cruise on a supported Linux host",
        )
        return
    if not path.is_dir() or not os.access(path, os.W_OK):
        _add_requirement(
            report,
            "shared-memory",
            False,
            code="unavailable-shared-memory-filesystem",
            expected="writable /dev/shm",
            observed="missing or not writable",
            remediation="mount a writable tmpfs at /dev/shm",
        )
        return
    try:
        free = shutil.disk_usage(path).free
    except OSError as exc:
        _add_requirement(
            report,
            "shared-memory",
            False,
            code="unavailable-shared-memory-filesystem",
            expected=f"at least {required} free bytes",
            observed=f"cannot query: {exc}",
            remediation="repair or remount /dev/shm, then rerun doctor",
        )
        return
    _add_requirement(
        report,
        "shared-memory",
        free >= required,
        code="insufficient-shared-memory",
        expected=f"at least {required} free bytes",
        observed=f"{free} free bytes",
        remediation="free /dev/shm space or enlarge the tmpfs mount",
    )


def _parse_npu_devices(output: str) -> dict[int, tuple[str, str]]:
    devices: dict[int, tuple[str, str]] = {}
    for match in re.finditer(
        r"^\|\s*(\d+)\s+(\S+)\s+\|\s+(OK|Warning|Alarm|Unknown)\b",
        output,
        re.MULTILINE | re.IGNORECASE,
    ):
        device_id, product, health = match.groups()
        devices[int(device_id)] = (product, health)
    return devices


def _parse_npu_processes(output: str) -> dict[int, tuple[int, ...]]:
    marker = re.search(r"Process\s+id", output, re.IGNORECASE)
    if marker is None:
        return {}
    processes: dict[int, list[int]] = {}
    for match in re.finditer(
        r"^\|\s*(\d+)\s+\d+\s+\|?\s*(\d+)\s+",
        output[marker.start() :],
        re.MULTILINE,
    ):
        device_id, process_id = (int(value) for value in match.groups())
        processes.setdefault(device_id, []).append(process_id)
    return {device: tuple(sorted(set(pids))) for device, pids in processes.items()}


def _check_npu(
    report: DoctorReport,
    profile: dict[str, Any],
    requirements: dict[str, Any],
    device_id: int,
) -> None:
    executable = shutil.which("npu-smi")
    if executable is None:
        remediation = "install the Ascend driver tools and put npu-smi on PATH"
        report.add(
            "npu-smi",
            "fail",
            _requirement_detail("npu-smi on PATH", "not found", remediation),
            code="missing-npu-management-tool",
            expected="npu-smi on PATH",
            observed="not found",
            remediation=remediation,
        )
        return
    try:
        result = _run([executable, "info"], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        observed = f"{type(exc).__name__}: {exc}"
        remediation = "repair driver access and rerun npu-smi info"
        report.add(
            "npu-smi",
            "fail",
            _requirement_detail("successful npu-smi info", observed, remediation),
            code="npu-management-query-failed",
            expected="successful npu-smi info",
            observed=observed,
            remediation=remediation,
        )
        return
    if result.returncode != 0:
        observed = f"exit={result.returncode}: {result.stderr.strip()}"
        remediation = "repair driver access and rerun npu-smi info"
        report.add(
            "npu-smi",
            "fail",
            _requirement_detail("exit=0", observed, remediation),
            code="npu-management-query-failed",
            expected="exit=0",
            observed=observed,
            remediation=remediation,
        )
        return
    version_match = re.search(r"npu-smi\s+(\S+)", result.stdout)
    observed_version = version_match.group(1) if version_match else "unknown"
    expected_version = profile["hardware"]["npu_smi_version"]
    _add_requirement(
        report,
        "npu-smi",
        observed_version == expected_version,
        code="incompatible-npu-smi-version",
        expected=expected_version,
        observed=observed_version,
        remediation=(
            f"install driver tools {expected_version} or select a profile validated "
            f"for {observed_version}"
        ),
    )
    driver_match = re.search(r"\bVersion:\s*(\S+)", result.stdout)
    observed_driver = driver_match.group(1) if driver_match else "unknown"
    expected_driver = profile["software"]["driver"]
    _add_requirement(
        report,
        "driver",
        observed_driver == expected_driver,
        code="incompatible-driver-version",
        expected=expected_driver,
        observed=observed_driver,
        remediation=(
            f"install driver {expected_driver} or qualify driver {observed_driver} "
            "as a new profile"
        ),
    )

    devices = _parse_npu_devices(result.stdout)
    allowed_products = {
        accelerator.split()[-1] for accelerator in requirements["accelerators"]
    }
    supported_count = sum(
        1 for product, _health in devices.values() if product in allowed_products
    )
    minimum_count = requirements["minimum_accelerator_count"]
    _add_requirement(
        report,
        "npu-count",
        supported_count >= minimum_count,
        code="insufficient-supported-accelerators",
        expected=f"at least {minimum_count} supported accelerator(s)",
        observed=f"{supported_count} supported; {len(devices)} total",
        remediation="select a host with enough supported Ascend accelerators",
    )

    expected_accelerator = profile["hardware"]["accelerator"].split()[-1]
    selected = devices.get(device_id)
    if selected is None:
        remediation = "choose a listed device with --device or repair NPU enumeration"
        report.add(
            "npu-device",
            "fail",
            _requirement_detail(
                f"device {device_id} listed as {expected_accelerator}",
                f"listed devices={sorted(devices)}",
                remediation,
            ),
            code="unavailable-npu-device",
            expected=f"device {device_id} listed as {expected_accelerator}",
            observed=f"listed devices={sorted(devices)}",
            remediation=remediation,
        )
        return
    product, health = selected
    _add_requirement(
        report,
        "npu-product",
        product in allowed_products and product == expected_accelerator,
        code="unsupported-accelerator",
        expected=f"one of {sorted(allowed_products)}; profile={expected_accelerator}",
        observed=product,
        remediation="use a supported accelerator or add a separately qualified profile",
    )
    _add_requirement(
        report,
        "npu-device",
        health.lower() == "ok",
        code="unhealthy-npu-device",
        expected="health=OK",
        observed=f"device={device_id} product={product} health={health}",
        remediation="resolve the NPU health alarm or select another healthy device",
    )
    active_pids = _parse_npu_processes(result.stdout).get(device_id, ())
    _add_requirement(
        report,
        "npu-availability",
        not active_pids,
        code="occupied-npu-device",
        expected="no active compute processes",
        observed=f"pids={list(active_pids)}" if active_pids else "idle",
        remediation="stop the listed workload or select an idle NPU",
    )


def run_npu_doctor(profile_id: str | None, device_id: int) -> DoctorReport:
    try:
        profile = get_compatibility_profile(profile_id)
    except CompatibilityError as exc:
        report = DoctorReport(mode="npu", profile=profile_id, checks=[])
        try:
            available = ", ".join(
                profile["id"] for profile in list_compatibility_profiles()
            )
        except CompatibilityError:
            available = "manifest unavailable"
        observed = profile_id if profile_id is not None else "not selected"
        remediation = f"select one of the packaged profiles: {available}"
        report.add(
            "compatibility-profile",
            "fail",
            _requirement_detail(available, observed, remediation),
            code="unknown-or-missing-compatibility-profile",
            expected=available,
            observed=observed,
            remediation=remediation,
        )
        return report
    report = DoctorReport(mode="npu", profile=profile["id"], checks=[])
    report.add("compatibility-profile", "pass", profile["id"])
    requirements = get_capability_requirements()
    architecture = platform.machine()
    allowed_architectures = requirements["architectures"]
    expected_architecture = profile["hardware"]["architecture"]
    _add_requirement(
        report,
        "architecture",
        architecture in allowed_architectures and architecture == expected_architecture,
        code="unsupported-host-architecture",
        expected=f"one of {allowed_architectures}; profile={expected_architecture}",
        observed=architecture,
        remediation="install Cruise on a supported aarch64 host",
    )
    _check_software(report, profile)
    cann_root = _check_cann(report, profile)
    _check_cann_capabilities(report, requirements, cann_root)
    _check_npu(report, profile, requirements, device_id)
    _check_shared_memory(report, requirements)
    profile_status = profile["status"]
    if profile_status.startswith("candidate-"):
        report.add(
            "profile-status",
            "warn",
            _requirement_detail(
                "validated profile",
                profile_status,
                "complete the profile qualification before claiming product support",
            ),
            code="unvalidated-compatibility-profile",
            expected="validated profile",
            observed=profile_status,
            remediation=(
                "complete the profile qualification before claiming product support"
            ),
        )
    report.add(
        "maturity",
        "warn",
        f"profile status={profile_status}; this is not Stable v1.0",
    )
    return report


def run_runtime_doctor(config: CruiseRuntimeConfig, *, deep: bool) -> DoctorReport:
    report = run_npu_doctor(config.compatibility_profile, config.device_id)
    report.mode = "runtime"
    try:
        config.validate_paths(deep=deep)
    except RuntimeConfigError as exc:
        report.add(
            "runtime-assets",
            "fail",
            str(exc),
            code=exc.code,
            expected=exc.expected,
            observed=exc.observed,
            remediation=exc.remediation,
        )
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
