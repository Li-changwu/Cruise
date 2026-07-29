from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from vllm_ascend_resident_epoch import cli
from vllm_ascend_resident_epoch import doctor as doctor_module
from vllm_ascend_resident_epoch import runtime_config as runtime_config_module
from vllm_ascend_resident_epoch.compatibility import (
    get_compatibility_profile,
    load_compatibility_manifest,
)
from vllm_ascend_resident_epoch.doctor import run_source_smoke
from vllm_ascend_resident_epoch.runtime_config import (
    CruiseRuntimeConfig,
    RuntimeConfigError,
    load_runtime_config,
)
from vllm_ascend_resident_epoch.version import (
    HOST_UDF_INPUTS,
    HOST_UDF_OUTPUTS,
    SIDECAR_PROTOCOL_VERSION,
    SIDECAR_REQUEST_BYTES,
    SIDECAR_RESPONSE_BYTES,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _runtime_fixture(tmp_path: Path) -> Path:
    assets = tmp_path / "assets"
    workspace = tmp_path / "controller"
    workspace.mkdir()
    server = _write(assets / "resident_epoch_server", b"server")
    server.chmod(0o755)
    air = _write(assets / "decoder.air", b"air")
    graph = _write(assets / "graph.json", "{}\n")
    function = _write(
        assets / "function.json",
        json.dumps(
            {
                "func_list": [{"func_name": "g4c_b4_resident_epoch"}],
                "input_num": HOST_UDF_INPUTS,
                "output_num": HOST_UDF_OUTPUTS,
                "workspace": str(workspace),
            }
        ),
    )
    tiling = _write(assets / "tiling.bin", b"tiling")
    resource = _write(assets / "resource.json", "{}\n")
    model_config = _write(assets / "config.json", "{}\n")
    model_index = _write(assets / "model.index.json", "{}\n")
    cann = _write(assets / "set_env.sh", "export CANN_TEST=1\n")

    weights = tmp_path / "weights"
    first = _write(weights / "first", b"first")
    second = _write(weights / "second", b"second")
    weight_records = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (first, second)
    ]
    weight_manifest = _write(
        assets / "weights.json",
        json.dumps({"files": weight_records}, sort_keys=True),
    )

    opp = tmp_path / "opp"
    for relative in ("op_impl", "op_proto", "op_api/lib"):
        (opp / relative).mkdir(parents=True)

    config = {
        "schema_version": 1,
        "compatibility_profile": "attempt74-910b2-cann851-r5",
        "device_id": 0,
        "cann_set_env": str(cann),
        "custom_opp_vendors": [str(opp)],
        "runtime": {
            "scratch_root": str(tmp_path / "scratch"),
            "minimum_scratch_free_bytes": 1,
            "max_steps": 4,
            "logical_capacity": 8,
            "startup_timeout_seconds": 30,
            "persistent_output_limit_bytes": 1024,
        },
        "assets": {
            "server": str(server),
            "air": str(air),
            "graph_config": str(graph),
            "function_config": str(function),
            "tiling": str(tiling),
            "external_weights": str(weights),
            "external_weights_manifest": str(weight_manifest),
            "resource_config": str(resource),
            "model_config": str(model_config),
            "model_index": str(model_index),
        },
        "integrity": {
            "air_sha256": _sha256(air),
            "graph_config_sha256": _sha256(graph),
            "tiling_sha256": _sha256(tiling),
            "external_weights_manifest_sha256": _sha256(weight_manifest),
            "external_weight_files": 2,
            "external_weight_bytes": first.stat().st_size + second.stat().st_size,
            "model_config_sha256": _sha256(model_config),
            "model_index_sha256": _sha256(model_index),
        },
    }
    config_path = tmp_path / "cruise.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def _load_test_runtime_config(tmp_path: Path, monkeypatch) -> CruiseRuntimeConfig:
    path = _runtime_fixture(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    integrity = value["integrity"]
    profile = {
        "id": value["compatibility_profile"],
        "assets": {
            "graph_config_sha256": integrity["graph_config_sha256"],
            "tiling_sha256": integrity["tiling_sha256"],
            "runtime_weights_manifest_sha256": integrity[
                "external_weights_manifest_sha256"
            ],
            "runtime_weight_files": integrity["external_weight_files"],
            "runtime_weight_bytes": integrity["external_weight_bytes"],
        },
        "model": {
            "config_sha256": integrity["model_config_sha256"],
            "index_sha256": integrity["model_index_sha256"],
        },
    }
    monkeypatch.setattr(
        runtime_config_module,
        "get_compatibility_profile",
        lambda profile_id: profile,
    )
    return load_runtime_config(path)


def test_compatibility_manifest_matches_versioned_contracts():
    manifest = load_compatibility_manifest()
    contracts = manifest["contracts"]
    assert contracts["sidecar_protocol"] == SIDECAR_PROTOCOL_VERSION
    assert contracts["sidecar_request_bytes"] == SIDECAR_REQUEST_BYTES
    assert contracts["sidecar_response_bytes"] == SIDECAR_RESPONSE_BYTES
    assert contracts["host_udf_inputs"] == HOST_UDF_INPUTS
    assert contracts["host_udf_outputs"] == HOST_UDF_OUTPUTS
    profile = get_compatibility_profile("attempt74-910b2-cann851-r5")
    assert profile["hardware"]["accelerator"] == "Ascend 910B2"
    assert profile["software"]["driver"] == "25.2.1"
    assert profile["software"]["cann"] == "8.5.1"


def test_npu_doctor_checks_driver_separately_from_npu_smi(monkeypatch):
    profile = get_compatibility_profile("attempt74-910b2-cann851-r5")
    output = """\
| npu-smi 25.2.1                   Version: 25.2.1 |
| 7     910B2                      | OK            |
"""
    monkeypatch.setattr(doctor_module, "get_compatibility_profile", lambda _: profile)
    monkeypatch.setattr(doctor_module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(doctor_module.shutil, "which", lambda _: "/usr/bin/npu-smi")
    monkeypatch.setattr(
        doctor_module,
        "_run",
        lambda command, timeout=30: subprocess.CompletedProcess(
            command, 0, stdout=output, stderr=""
        ),
    )
    monkeypatch.setattr(doctor_module, "_check_software", lambda report, value: None)
    monkeypatch.setattr(doctor_module, "_check_cann", lambda report, value: None)

    report = doctor_module.run_npu_doctor(profile["id"], 7)
    checks = {check.name: check for check in report.checks}
    assert checks["npu-smi"].status == "pass"
    assert checks["driver"].status == "pass"
    assert checks["driver"].detail == "expected=25.2.1 observed=25.2.1"


def test_native_protocol_header_matches_python_constants():
    header = (ROOT / "native" / "resident_epoch_protocol.h").read_text(
        encoding="utf-8"
    )
    assert f"CRUISE_SIDECAR_PROTOCOL_VERSION {SIDECAR_PROTOCOL_VERSION}" in header
    assert f"CRUISE_SIDECAR_REQUEST_BYTES {SIDECAR_REQUEST_BYTES}" in header
    assert f"CRUISE_SIDECAR_RESPONSE_BYTES {SIDECAR_RESPONSE_BYTES}" in header
    server = (ROOT / "native" / "resident_epoch_server.cpp").read_text(
        encoding="utf-8"
    )
    assert '#include "resident_epoch_protocol.h"' in server
    assert "constexpr uint16_t kProtocolVersion = 3" not in server


def test_example_runtime_config_is_structurally_valid():
    config = load_runtime_config(ROOT / "config" / "cruise.example.json")
    profile = get_compatibility_profile(config.compatibility_profile)
    assert config.compatibility_profile == "attempt74-910b2-cann851-r5"
    assert config.runtime.max_steps == 8
    assert config.integrity.external_weight_files == 342
    assert config.integrity.air_sha256 != profile["assets"]["frozen_air_sha256"]


def test_runtime_config_rejects_unknown_fields(tmp_path):
    value = json.loads(
        (ROOT / "config" / "cruise.example.json").read_text(encoding="utf-8")
    )
    value["typo"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeConfigError, match="unknown top-level fields: typo"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "field",
    [
        "graph_config_sha256",
        "tiling_sha256",
        "external_weights_manifest_sha256",
        "external_weight_files",
        "external_weight_bytes",
        "model_config_sha256",
        "model_index_sha256",
    ],
)
def test_runtime_config_rejects_profile_identity_mismatch(tmp_path, field):
    value = json.loads(
        (ROOT / "config" / "cruise.example.json").read_text(encoding="utf-8")
    )
    current = value["integrity"][field]
    value["integrity"][field] = current + 1 if isinstance(current, int) else "0" * 64
    path = tmp_path / "profile-mismatch.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        RuntimeConfigError,
        match=rf"runtime assets do not match compatibility profile .*integrity\.{field}",
    ):
        load_runtime_config(path)


def test_runtime_config_validates_paths_hashes_and_weight_manifest(
    tmp_path, monkeypatch
):
    config = _load_test_runtime_config(tmp_path, monkeypatch)
    monkeypatch.setattr(CruiseRuntimeConfig, "_validate_device_paths", lambda self: None)
    config.validate_paths(deep=True)

    (config.assets.external_weights / "first").write_bytes(b"changed")
    with pytest.raises(RuntimeConfigError, match="external weight byte count mismatch"):
        config.validate_paths(deep=False)


def test_runtime_environment_is_complete(tmp_path, monkeypatch):
    config = _load_test_runtime_config(tmp_path, monkeypatch)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    environment = config.environment(run_directory, {"PATH": "/bin"})
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "0"
    assert environment["VLLM_ASCEND_RESIDENT_EPOCH_STEPS"] == "4"
    assert environment["VLLM_ASCEND_RESIDENT_EPOCH_CAPACITY"] == "8"
    assert environment["VLLM_ASCEND_RESIDENT_EPOCH_SOCKET"].endswith(
        "resident-epoch.sock"
    )
    assert environment["VLLM_ASCEND_RESIDENT_EPOCH_BACKEND_FACTORY"].endswith(
        ":create_sidecar_engine"
    )
    assert str(config.custom_opp_vendors[0]) in environment["ASCEND_CUSTOM_OPP_PATH"]


def test_no_npu_smoke_and_cli_exit_success(capsys):
    report = run_source_smoke()
    assert report.passed
    assert cli.main(["smoke", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["pass"] is True
    assert output["mode"] == "source-smoke"


def test_cleanup_only_removes_empty_marked_scratch(tmp_path, monkeypatch):
    config = _load_test_runtime_config(tmp_path, monkeypatch)
    config.runtime.scratch_root.mkdir()
    marker = config.runtime.scratch_root / cli._SCRATCH_MARKER
    marker.write_text(cli._MARKER_CONTENT, encoding="ascii")
    unknown = config.runtime.scratch_root / "unknown"
    unknown.write_text("keep", encoding="ascii")
    with pytest.raises(RuntimeConfigError, match="scratch_root is not empty"):
        cli._remove_empty_scratch_root(config)
    assert unknown.is_file()
    unknown.unlink()
    cli._remove_empty_scratch_root(config)
    assert not config.runtime.scratch_root.exists()
