from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .compatibility import get_compatibility_profile
from .version import (
    HOST_UDF_INPUTS,
    HOST_UDF_OUTPUTS,
    RUNTIME_CONFIG_SCHEMA_VERSION,
)


class RuntimeConfigError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        expected: str | None = None,
        observed: str | None = None,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.expected = expected
        self.observed = observed
        self.remediation = remediation


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TOP_LEVEL = {
    "schema_version",
    "compatibility_profile",
    "device_id",
    "cann_set_env",
    "custom_opp_vendors",
    "runtime",
    "assets",
    "integrity",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"{name} must be an object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeConfigError(f"{name} must be a positive integer")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeConfigError(f"{name} must be a lowercase SHA256")
    return value


def _path(value: Any, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _require_keys(record: dict[str, Any], required: set[str], name: str) -> None:
    missing = sorted(required - set(record))
    if missing:
        raise RuntimeConfigError(f"{name} is missing: {', '.join(missing)}")


@dataclass(frozen=True)
class RuntimeSettings:
    scratch_root: Path
    minimum_scratch_free_bytes: int
    max_steps: int
    logical_capacity: int
    startup_timeout_seconds: int
    persistent_output_limit_bytes: int


@dataclass(frozen=True)
class RuntimeAssets:
    server: Path
    air: Path
    graph_config: Path
    function_config: Path
    tiling: Path
    external_weights: Path
    external_weights_manifest: Path
    resource_config: Path
    model_config: Path
    model_index: Path


@dataclass(frozen=True)
class AssetIntegrity:
    air_sha256: str
    graph_config_sha256: str
    tiling_sha256: str
    external_weights_manifest_sha256: str
    external_weight_files: int
    external_weight_bytes: int
    model_config_sha256: str
    model_index_sha256: str


def _validate_profile_identity(
    profile: dict[str, Any], integrity: AssetIntegrity
) -> None:
    profile_id = profile["id"]
    assets = _mapping(profile.get("assets"), f"profile {profile_id} assets")
    model = _mapping(profile.get("model"), f"profile {profile_id} model")
    bindings = {
        "integrity.graph_config_sha256": (
            integrity.graph_config_sha256,
            assets.get("graph_config_sha256"),
        ),
        "integrity.tiling_sha256": (
            integrity.tiling_sha256,
            assets.get("tiling_sha256"),
        ),
        "integrity.external_weights_manifest_sha256": (
            integrity.external_weights_manifest_sha256,
            assets.get("runtime_weights_manifest_sha256"),
        ),
        "integrity.external_weight_files": (
            integrity.external_weight_files,
            assets.get("runtime_weight_files"),
        ),
        "integrity.external_weight_bytes": (
            integrity.external_weight_bytes,
            assets.get("runtime_weight_bytes"),
        ),
        "integrity.model_config_sha256": (
            integrity.model_config_sha256,
            model.get("config_sha256"),
        ),
        "integrity.model_index_sha256": (
            integrity.model_index_sha256,
            model.get("index_sha256"),
        ),
    }
    mismatches = [
        f"{name} expected {expected!r}, got {actual!r}"
        for name, (actual, expected) in bindings.items()
        if actual != expected
    ]
    if mismatches:
        raise RuntimeConfigError(
            f"runtime assets do not match compatibility profile {profile_id}: "
            + "; ".join(mismatches)
        )


@dataclass(frozen=True)
class CruiseRuntimeConfig:
    source: Path
    compatibility_profile: str
    device_id: int
    cann_set_env: Path
    custom_opp_vendors: tuple[Path, ...]
    runtime: RuntimeSettings
    assets: RuntimeAssets
    integrity: AssetIntegrity

    def validate_paths(self, *, deep: bool = False) -> None:
        required_files = {
            "cann_set_env": self.cann_set_env,
            "server": self.assets.server,
            "air": self.assets.air,
            "graph_config": self.assets.graph_config,
            "function_config": self.assets.function_config,
            "tiling": self.assets.tiling,
            "external_weights_manifest": self.assets.external_weights_manifest,
            "resource_config": self.assets.resource_config,
            "model_config": self.assets.model_config,
            "model_index": self.assets.model_index,
        }
        for name, path in required_files.items():
            if not path.is_file():
                raise RuntimeConfigError(
                    f"{name} is not a file: {path}",
                    code="missing-runtime-asset",
                    expected=f"regular file for {name} at {path}",
                    observed="missing or not a regular file",
                    remediation=(
                        "provision the content-addressed asset bundle declared by "
                        f"profile {self.compatibility_profile}; do not substitute an "
                        "unqualified model or artifact"
                    ),
                )
        if os.name == "posix" and not self.assets.server.stat().st_mode & 0o111:
            raise RuntimeConfigError(
                f"server is not executable: {self.assets.server}",
                code="invalid-runtime-asset-permissions",
                expected=f"executable server at {self.assets.server}",
                observed="regular file without an executable bit",
                remediation="restore the executable mode from the qualified asset bundle",
            )
        if not self.assets.external_weights.is_dir():
            raise RuntimeConfigError(
                f"external_weights is not a directory: {self.assets.external_weights}",
                code="missing-runtime-asset",
                expected=(
                    "external weight directory with "
                    f"{self.integrity.external_weight_files} files and "
                    f"{self.integrity.external_weight_bytes} bytes at "
                    f"{self.assets.external_weights}"
                ),
                observed="missing or not a directory",
                remediation=(
                    "provision the content-addressed runtime weights declared by "
                    f"profile {self.compatibility_profile}; do not download or reuse a "
                    "different model revision"
                ),
            )

        self._validate_device_paths()
        self._validate_opp_vendors()
        self._validate_function_config()
        self._validate_hashes()
        self._validate_external_weights(deep=deep)

    def _validate_device_paths(self) -> None:
        if os.name != "posix":
            return
        scratch = self.runtime.scratch_root.resolve()
        weights = self.assets.external_weights.resolve()
        try:
            scratch.relative_to("/dev/shm")
        except ValueError as exc:
            raise RuntimeConfigError(
                f"scratch_root must be below /dev/shm: {scratch}"
            ) from exc
        if scratch == Path("/dev/shm"):
            raise RuntimeConfigError("scratch_root must not be /dev/shm itself")
        try:
            weights.relative_to(scratch)
        except ValueError:
            pass
        else:
            raise RuntimeConfigError(
                "external_weights must not be inside the cleanup-managed scratch_root"
            )

    def _validate_opp_vendors(self) -> None:
        for vendor in self.custom_opp_vendors:
            for relative in ("op_impl", "op_proto", "op_api/lib"):
                path = vendor / relative
                if not path.is_dir():
                    raise RuntimeConfigError(f"custom OPP directory is missing: {path}")

    def _validate_function_config(self) -> None:
        try:
            value = json.loads(self.assets.function_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeConfigError(f"cannot parse function_config: {exc}") from exc
        functions = value.get("func_list")
        if functions != [{"func_name": "g4c_b4_resident_epoch"}]:
            raise RuntimeConfigError("function_config selects an unexpected UDF")
        if value.get("input_num") != HOST_UDF_INPUTS:
            raise RuntimeConfigError("function_config Host-UDF input count is incompatible")
        if value.get("output_num") != HOST_UDF_OUTPUTS:
            raise RuntimeConfigError("function_config Host-UDF output count is incompatible")
        workspace = value.get("workspace")
        if not isinstance(workspace, str) or not Path(workspace).is_dir():
            raise RuntimeConfigError("function_config workspace is missing")

    def _validate_hashes(self) -> None:
        expected = {
            self.assets.air: self.integrity.air_sha256,
            self.assets.graph_config: self.integrity.graph_config_sha256,
            self.assets.tiling: self.integrity.tiling_sha256,
            self.assets.external_weights_manifest: (
                self.integrity.external_weights_manifest_sha256
            ),
            self.assets.model_config: self.integrity.model_config_sha256,
            self.assets.model_index: self.integrity.model_index_sha256,
        }
        for path, digest in expected.items():
            observed = sha256_file(path)
            if observed != digest:
                raise RuntimeConfigError(
                    f"SHA256 mismatch for {path}: expected {digest}, observed {observed}"
                )

    def _validate_external_weights(self, *, deep: bool) -> None:
        files = sorted(path for path in self.assets.external_weights.iterdir() if path.is_file())
        observed_bytes = sum(path.stat().st_size for path in files)
        if len(files) != self.integrity.external_weight_files:
            raise RuntimeConfigError(
                "external weight file count mismatch: "
                f"expected {self.integrity.external_weight_files}, observed {len(files)}"
            )
        if observed_bytes != self.integrity.external_weight_bytes:
            raise RuntimeConfigError(
                "external weight byte count mismatch: "
                f"expected {self.integrity.external_weight_bytes}, observed {observed_bytes}"
            )
        if not deep:
            return

        try:
            manifest = json.loads(
                self.assets.external_weights_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeConfigError(f"cannot parse external weight manifest: {exc}") from exc
        records = manifest.get("files")
        if not isinstance(records, list) or len(records) != len(files):
            raise RuntimeConfigError("external weight manifest file list is incomplete")
        expected_names = {path.name for path in files}
        manifest_names = {record.get("name") for record in records if isinstance(record, dict)}
        if manifest_names != expected_names:
            raise RuntimeConfigError("external weight manifest names do not match directory")
        for record in records:
            name = record["name"]
            path = self.assets.external_weights / name
            if path.stat().st_size != record.get("bytes"):
                raise RuntimeConfigError(f"external weight size mismatch: {name}")
            if sha256_file(path) != record.get("sha256"):
                raise RuntimeConfigError(f"external weight SHA256 mismatch: {name}")

    def environment(self, run_directory: Path, base: Mapping[str, str]) -> dict[str, str]:
        run_directory = run_directory.resolve()
        environment = dict(base)
        cache = run_directory / "cache"
        ascend_cache = cache / "ascend"
        torchinductor_cache = cache / "torchinductor"
        triton_cache = cache / "triton"
        xdg_cache = cache / "xdg"
        logs = run_directory / "logs"
        temporary = run_directory / "tmp"
        graph_external_weights = run_directory / "graph-external-weights"
        for path in (
            ascend_cache,
            torchinductor_cache,
            triton_cache,
            xdg_cache,
            logs,
            temporary,
            graph_external_weights,
        ):
            path.mkdir(parents=True, exist_ok=True)

        environment.update(
            {
                "ASCEND_RT_VISIBLE_DEVICES": str(self.device_id),
                "ASCEND_GLOBAL_LOG_LEVEL": "3",
                "ASCEND_SLOG_PRINT_TO_STDOUT": "1",
                "ASCEND_PROCESS_LOG_PATH": str(logs),
                "ASCEND_CACHE_PATH": str(ascend_cache),
                "TMPDIR": str(temporary),
                "TORCHINDUCTOR_CACHE_DIR": str(torchinductor_cache),
                "TRITON_CACHE_DIR": str(triton_cache),
                "XDG_CACHE_HOME": str(xdg_cache),
                "PYTHONDONTWRITEBYTECODE": "1",
                "RESOURCE_CONFIG_PATH": str(self.assets.resource_config),
                "VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY": (
                    "vllm_ascend_resident_epoch.sidecar_backend:create_sidecar_engine"
                ),
                "VLLM_ASCEND_RESIDENT_EPOCH_SERVER": str(self.assets.server),
                "VLLM_ASCEND_RESIDENT_EPOCH_AIR": str(self.assets.air),
                "VLLM_ASCEND_RESIDENT_EPOCH_GRAPH_CONFIG": str(
                    self.assets.graph_config
                ),
                "VLLM_ASCEND_RESIDENT_EPOCH_FUNC_CONFIG": str(
                    self.assets.function_config
                ),
                "VLLM_ASCEND_RESIDENT_EPOCH_TILING": str(self.assets.tiling),
                "VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS": str(
                    graph_external_weights
                ),
                "VLLM_ASCEND_RESIDENT_EPOCH_RUNTIME_WEIGHTS": str(
                    self.assets.external_weights
                ),
                "VLLM_ASCEND_RESIDENT_EPOCH_SOCKET": str(
                    run_directory / "resident-epoch.sock"
                ),
                "VLLM_ASCEND_RESIDENT_EPOCH_STARTUP_TIMEOUT": str(
                    self.runtime.startup_timeout_seconds
                ),
                "VLLM_ASCEND_RESIDENT_EPOCH_STEPS": str(self.runtime.max_steps),
                "VLLM_ASCEND_RESIDENT_EPOCH_CAPACITY": str(
                    self.runtime.logical_capacity
                ),
            }
        )
        if self.custom_opp_vendors:
            vendors = ":".join(str(path) for path in self.custom_opp_vendors)
            existing_opp = environment.get("ASCEND_CUSTOM_OPP_PATH")
            environment["ASCEND_CUSTOM_OPP_PATH"] = (
                f"{vendors}:{existing_opp}" if existing_opp else vendors
            )
            libraries = ":".join(
                str(path / "op_api" / "lib") for path in self.custom_opp_vendors
            )
            existing_library = environment.get("LD_LIBRARY_PATH")
            environment["LD_LIBRARY_PATH"] = (
                f"{libraries}:{existing_library}" if existing_library else libraries
            )
        return environment


def load_runtime_config(path: str | Path) -> CruiseRuntimeConfig:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeConfigError(f"configuration does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeConfigError(f"configuration is not valid JSON: {exc}") from exc
    root = _mapping(value, "configuration")
    unknown = sorted(set(root) - _ALLOWED_TOP_LEVEL)
    if unknown:
        raise RuntimeConfigError(f"unknown top-level fields: {', '.join(unknown)}")
    _require_keys(root, _ALLOWED_TOP_LEVEL, "configuration")
    if root["schema_version"] != RUNTIME_CONFIG_SCHEMA_VERSION:
        raise RuntimeConfigError("unsupported runtime configuration schema")

    profile_id = root["compatibility_profile"]
    if not isinstance(profile_id, str):
        raise RuntimeConfigError("compatibility_profile must be a string")
    profile = get_compatibility_profile(profile_id)
    device_id = root["device_id"]
    if not isinstance(device_id, int) or isinstance(device_id, bool) or device_id < 0:
        raise RuntimeConfigError("device_id must be a non-negative integer")

    base = source.parent
    runtime_value = _mapping(root["runtime"], "runtime")
    runtime_required = {
        "scratch_root",
        "minimum_scratch_free_bytes",
        "max_steps",
        "logical_capacity",
        "startup_timeout_seconds",
        "persistent_output_limit_bytes",
    }
    _require_keys(runtime_value, runtime_required, "runtime")
    if set(runtime_value) != runtime_required:
        raise RuntimeConfigError("runtime contains unknown fields")
    max_steps = _positive_int(runtime_value["max_steps"], "runtime.max_steps")
    if max_steps not in (1, 2, 4, 8):
        raise RuntimeConfigError("runtime.max_steps must be one of 1, 2, 4, 8")
    logical_capacity = _positive_int(
        runtime_value["logical_capacity"], "runtime.logical_capacity"
    )
    if logical_capacity < max_steps:
        raise RuntimeConfigError("runtime.logical_capacity must cover max_steps")
    runtime = RuntimeSettings(
        scratch_root=_path(runtime_value["scratch_root"], "runtime.scratch_root", base),
        minimum_scratch_free_bytes=_positive_int(
            runtime_value["minimum_scratch_free_bytes"],
            "runtime.minimum_scratch_free_bytes",
        ),
        max_steps=max_steps,
        logical_capacity=logical_capacity,
        startup_timeout_seconds=_positive_int(
            runtime_value["startup_timeout_seconds"],
            "runtime.startup_timeout_seconds",
        ),
        persistent_output_limit_bytes=_positive_int(
            runtime_value["persistent_output_limit_bytes"],
            "runtime.persistent_output_limit_bytes",
        ),
    )

    asset_value = _mapping(root["assets"], "assets")
    asset_fields = {
        "server",
        "air",
        "graph_config",
        "function_config",
        "tiling",
        "external_weights",
        "external_weights_manifest",
        "resource_config",
        "model_config",
        "model_index",
    }
    _require_keys(asset_value, asset_fields, "assets")
    if set(asset_value) != asset_fields:
        raise RuntimeConfigError("assets contains unknown fields")
    assets = RuntimeAssets(
        **{
            name: _path(asset_value[name], f"assets.{name}", base)
            for name in asset_fields
        }
    )

    integrity_value = _mapping(root["integrity"], "integrity")
    integrity_fields = {
        "air_sha256",
        "graph_config_sha256",
        "tiling_sha256",
        "external_weights_manifest_sha256",
        "external_weight_files",
        "external_weight_bytes",
        "model_config_sha256",
        "model_index_sha256",
    }
    _require_keys(integrity_value, integrity_fields, "integrity")
    if set(integrity_value) != integrity_fields:
        raise RuntimeConfigError("integrity contains unknown fields")
    integrity = AssetIntegrity(
        air_sha256=_sha256(integrity_value["air_sha256"], "integrity.air_sha256"),
        graph_config_sha256=_sha256(
            integrity_value["graph_config_sha256"],
            "integrity.graph_config_sha256",
        ),
        tiling_sha256=_sha256(
            integrity_value["tiling_sha256"], "integrity.tiling_sha256"
        ),
        external_weights_manifest_sha256=_sha256(
            integrity_value["external_weights_manifest_sha256"],
            "integrity.external_weights_manifest_sha256",
        ),
        external_weight_files=_positive_int(
            integrity_value["external_weight_files"],
            "integrity.external_weight_files",
        ),
        external_weight_bytes=_positive_int(
            integrity_value["external_weight_bytes"],
            "integrity.external_weight_bytes",
        ),
        model_config_sha256=_sha256(
            integrity_value["model_config_sha256"],
            "integrity.model_config_sha256",
        ),
        model_index_sha256=_sha256(
            integrity_value["model_index_sha256"],
            "integrity.model_index_sha256",
        ),
    )
    _validate_profile_identity(profile, integrity)

    opp_value = root["custom_opp_vendors"]
    if not isinstance(opp_value, list) or not opp_value:
        raise RuntimeConfigError("custom_opp_vendors must be a non-empty list")
    opp_vendors = tuple(
        _path(value, f"custom_opp_vendors[{index}]", base)
        for index, value in enumerate(opp_value)
    )
    if len(set(opp_vendors)) != len(opp_vendors):
        raise RuntimeConfigError("custom_opp_vendors contains duplicates")

    return CruiseRuntimeConfig(
        source=source,
        compatibility_profile=profile_id,
        device_id=device_id,
        cann_set_env=_path(root["cann_set_env"], "cann_set_env", base),
        custom_opp_vendors=opp_vendors,
        runtime=runtime,
        assets=assets,
        integrity=integrity,
    )
