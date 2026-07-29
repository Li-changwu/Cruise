from __future__ import annotations

from importlib.resources import files
import json
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
        assets = profile.get("assets")
        if not isinstance(assets, dict):
            raise CompatibilityError(f"profile {profile_id} has no asset record")
        for name, value in assets.items():
            if name.endswith("_sha256") and (
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
            ):
                raise CompatibilityError(
                    f"profile {profile_id} has invalid {name}"
                )


def get_compatibility_profile(profile_id: str | None = None) -> dict[str, Any]:
    manifest = load_compatibility_manifest()
    profiles = manifest["profiles"]
    if profile_id is None:
        if len(profiles) != 1:
            raise CompatibilityError("a compatibility profile must be selected")
        return profiles[0]
    for profile in profiles:
        if profile["id"] == profile_id:
            return profile
    raise CompatibilityError(f"unknown compatibility profile {profile_id!r}")

