from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Any

from .version import (
    COMPATIBILITY_SCHEMA_VERSION,
    DECODER_ABI_VERSION,
    DECODER_INPUTS,
    DECODER_OUTPUTS,
    HOST_UDF_ABI_VERSION,
    HOST_UDF_INPUTS,
    HOST_UDF_OUTPUTS,
    PACKAGE_VERSION,
    PRODUCT_MATURITY,
    PYTHON_CONTRACT_VERSION,
    RUNTIME_CONFIG_SCHEMA_VERSION,
    SIDECAR_PROTOCOL_VERSION,
    SIDECAR_REQUEST_BYTES,
    SIDECAR_RESPONSE_BYTES,
)


class CompatibilityError(ValueError):
    pass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _non_empty_strings(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise CompatibilityError(f"{name} must be a non-empty string list")
    if len(set(value)) != len(value):
        raise CompatibilityError(f"{name} contains duplicates")
    return value


def load_compatibility_manifest() -> dict[str, Any]:
    resource = files(__package__).joinpath("compatibility.json")
    manifest = json.loads(resource.read_text(encoding="utf-8"))
    validate_compatibility_manifest(manifest)
    return manifest


def validate_compatibility_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != COMPATIBILITY_SCHEMA_VERSION:
        raise CompatibilityError("unsupported compatibility manifest schema")

    product = manifest.get("product")
    if not isinstance(product, dict):
        raise CompatibilityError("compatibility manifest has no product record")
    expected_product = {
        "name": "Cruise",
        "package": "vllm-ascend-resident-epoch",
        "version": PACKAGE_VERSION,
        "maturity": PRODUCT_MATURITY,
    }
    if product != expected_product:
        raise CompatibilityError("compatibility product record disagrees with package")

    expected_contracts = {
        "python_contract": PYTHON_CONTRACT_VERSION,
        "runtime_config": RUNTIME_CONFIG_SCHEMA_VERSION,
        "sidecar_protocol": SIDECAR_PROTOCOL_VERSION,
        "sidecar_request_bytes": SIDECAR_REQUEST_BYTES,
        "sidecar_response_bytes": SIDECAR_RESPONSE_BYTES,
        "host_udf_abi": HOST_UDF_ABI_VERSION,
        "host_udf_inputs": HOST_UDF_INPUTS,
        "host_udf_outputs": HOST_UDF_OUTPUTS,
        "decoder_abi": DECODER_ABI_VERSION,
        "decoder_inputs": DECODER_INPUTS,
        "decoder_outputs": DECODER_OUTPUTS,
    }
    if manifest.get("contracts") != expected_contracts:
        raise CompatibilityError("compatibility contract versions disagree with code")

    requirements = manifest.get("capability_requirements")
    if not isinstance(requirements, dict):
        raise CompatibilityError("compatibility manifest has no capability requirements")
    _non_empty_strings(
        requirements.get("architectures"),
        "capability_requirements.architectures",
    )
    _non_empty_strings(
        requirements.get("accelerators"),
        "capability_requirements.accelerators",
    )
    _non_empty_strings(
        requirements.get("required_python_modules"),
        "capability_requirements.required_python_modules",
    )
    required_python_symbols = requirements.get("required_python_symbols")
    if not isinstance(required_python_symbols, list) or not required_python_symbols:
        raise CompatibilityError(
            "capability_requirements.required_python_symbols must be a non-empty list"
        )
    symbol_ids: set[str] = set()
    for symbol in required_python_symbols:
        if not isinstance(symbol, dict) or set(symbol) != {
            "id",
            "alternatives",
            "remediation",
        }:
            raise CompatibilityError(
                "each required Python symbol must contain id, alternatives, and "
                "remediation"
            )
        if any(
            not isinstance(symbol[name], str) or not symbol[name].strip()
            for name in ("id", "remediation")
        ):
            raise CompatibilityError("required Python symbol fields must be non-empty")
        alternatives = symbol["alternatives"]
        if not isinstance(alternatives, list) or not alternatives:
            raise CompatibilityError(
                "required Python symbol alternatives must be a non-empty list"
            )
        for alternative in alternatives:
            if not isinstance(alternative, dict) or set(alternative) != {
                "module",
                "attribute",
            }:
                raise CompatibilityError(
                    "each Python symbol alternative must contain module and attribute"
                )
            if any(
                not isinstance(alternative[name], str)
                or not alternative[name].strip()
                for name in ("module", "attribute")
            ):
                raise CompatibilityError(
                    "Python symbol alternative fields must be non-empty"
                )
        if symbol["id"] in symbol_ids:
            raise CompatibilityError("required Python symbols contain duplicate ids")
        symbol_ids.add(symbol["id"])
    for name in ("minimum_accelerator_count", "minimum_shared_memory_free_bytes"):
        value = requirements.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise CompatibilityError(
                f"capability_requirements.{name} must be a positive integer"
            )
    required_cann_files = requirements.get("required_cann_files")
    if not isinstance(required_cann_files, list) or not required_cann_files:
        raise CompatibilityError(
            "capability_requirements.required_cann_files must be a non-empty list"
        )
    component_ids: set[str] = set()
    component_names: set[str] = set()
    for component in required_cann_files:
        if not isinstance(component, dict) or set(component) != {
            "id",
            "name",
            "relative_paths",
        }:
            raise CompatibilityError(
                "each required CANN file must contain id, name, and relative_paths"
            )
        component_id = component["id"]
        component_name = component["name"]
        if not isinstance(component_id, str) or not component_id.strip():
            raise CompatibilityError("required CANN file id must be non-empty")
        if not isinstance(component_name, str) or not component_name.strip():
            raise CompatibilityError("required CANN file name must be non-empty")
        if "/" in component_name or "\\" in component_name:
            raise CompatibilityError("required CANN file name must be a basename")
        relative_paths = _non_empty_strings(
            component["relative_paths"],
            f"required CANN file {component_id} relative_paths",
        )
        if any(Path(value).is_absolute() or ".." in Path(value).parts for value in relative_paths):
            raise CompatibilityError(
                f"required CANN file {component_id} has an unsafe relative path"
            )
        if any(Path(value).name != component_name for value in relative_paths):
            raise CompatibilityError(
                f"required CANN file {component_id} paths disagree with its name"
            )
        if component_id in component_ids or component_name in component_names:
            raise CompatibilityError("required CANN files contain duplicates")
        component_ids.add(component_id)
        component_names.add(component_name)

    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise CompatibilityError("compatibility manifest has no profiles")
    profile_ids: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict) or not isinstance(profile.get("id"), str):
            raise CompatibilityError("compatibility profile has no id")
        profile_id = profile["id"]
        if profile_id in profile_ids:
            raise CompatibilityError(f"duplicate compatibility profile {profile_id}")
        profile_ids.add(profile_id)
        software = profile.get("software")
        if not isinstance(software, dict) or not isinstance(
            software.get("driver"), str
        ):
            raise CompatibilityError(
                f"profile {profile_id} has no driver version"
            )
        assets = profile.get("assets")
        if not isinstance(assets, dict):
            raise CompatibilityError(f"profile {profile_id} has no asset record")
        required_assets = {
            "graph_config_sha256",
            "tiling_sha256",
            "runtime_weights_manifest_sha256",
            "runtime_weight_files",
            "runtime_weight_bytes",
        }
        missing_assets = sorted(required_assets - set(assets))
        if missing_assets:
            raise CompatibilityError(
                f"profile {profile_id} is missing assets: "
                + ", ".join(missing_assets)
            )
        for name, value in assets.items():
            if name.endswith("_sha256") and (
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
            ):
                raise CompatibilityError(
                    f"profile {profile_id} has invalid {name}"
                )
        for name in ("runtime_weight_files", "runtime_weight_bytes"):
            value = assets[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise CompatibilityError(
                    f"profile {profile_id} has invalid {name}"
                )
        model = profile.get("model")
        if not isinstance(model, dict):
            raise CompatibilityError(f"profile {profile_id} has no model record")
        for name in ("config_sha256", "index_sha256"):
            value = model.get(name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise CompatibilityError(
                    f"profile {profile_id} has invalid model {name}"
                )

        hardware = profile.get("hardware")
        if not isinstance(hardware, dict):
            raise CompatibilityError(f"profile {profile_id} has no hardware record")
        if hardware.get("architecture") not in requirements["architectures"]:
            raise CompatibilityError(
                f"profile {profile_id} uses an unsupported architecture"
            )
        if hardware.get("accelerator") not in requirements["accelerators"]:
            raise CompatibilityError(
                f"profile {profile_id} uses an unsupported accelerator"
            )
        accelerator_count = hardware.get("accelerator_count")
        if (
            not isinstance(accelerator_count, int)
            or isinstance(accelerator_count, bool)
            or accelerator_count < requirements["minimum_accelerator_count"]
        ):
            raise CompatibilityError(
                f"profile {profile_id} has insufficient accelerator_count"
            )


def get_capability_requirements() -> dict[str, Any]:
    return load_compatibility_manifest()["capability_requirements"]


def list_compatibility_profiles() -> tuple[dict[str, Any], ...]:
    return tuple(load_compatibility_manifest()["profiles"])


def get_compatibility_profile(profile_id: str | None = None) -> dict[str, Any]:
    manifest = load_compatibility_manifest()
    profiles = manifest["profiles"]
    if profile_id is None:
        if len(profiles) != 1:
            available = ", ".join(profile["id"] for profile in profiles)
            raise CompatibilityError(
                "a compatibility profile must be selected; "
                f"available profiles: {available}"
            )
        return profiles[0]
    for profile in profiles:
        if profile["id"] == profile_id:
            return profile
    raise CompatibilityError(f"unknown compatibility profile {profile_id!r}")
