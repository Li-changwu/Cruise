from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "m1_batched_prefill" / "run_on_910b2.sh"


def test_m1_runner_reuses_content_addressed_runtime_weights() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "runtime_asset_root=${CRUISE_RUNTIME_ASSET_ROOT:-}" in script
    assert (
        "runtime_weights=${runtime_asset_root}/runtime-weights/"
        "${runtime_weight_digest}"
    ) in script
    assert "--persistent-asset-root \"${runtime_asset_root}\"" in script
    assert "VLLM_ASCEND_RESIDENT_EPOCH_RUNTIME_WEIGHTS=${runtime_weights}" in script


def test_m1_runner_keeps_mutable_graph_weights_in_scratch() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "external_weights=${scratch}/external-weights" in script
    assert "VLLM_ASCEND_RESIDENT_EPOCH_EXTERNAL_WEIGHTS=${external_weights}" in script
    assert 'storage_guard_assert_scratch_path "${runtime_weights}"' in script
    assert 'if [[ -z "${runtime_asset_root}" ]]' in script


def test_m1_runner_generates_resource_config_for_selected_physical_npu() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "resource_config_override=${CRUISE_RESOURCE_CONFIG:-}" in script
    assert "${scratch}/numa_config.physical${physical_npu}.json" in script
    assert '--physical-npu "${physical_npu}"' in script
    assert '--deploy-root "${deploy_root}"' in script
    assert "deploy_root=${scratch}/dataflow-deploy" in script
    assert "RESOURCE_CONFIG_PATH=${resource_config}" in script


def test_m1_runner_bounds_stock_vllm_kv_memory() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "vllm_kv_cache_bytes=${CRUISE_VLLM_KV_CACHE_BYTES:-536870912}" in script
    assert "CRUISE_VLLM_KV_CACHE_BYTES=${vllm_kv_cache_bytes}" in script


def test_m1_runner_defaults_api_tokenizer_to_frozen_model() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "CRUISE_API_TOKENIZER=${CRUISE_API_TOKENIZER:-${model}}" in script


def test_m1_runner_passes_manifest_to_comparison() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert '--mode compare --cases "${case_manifest}"' in script
