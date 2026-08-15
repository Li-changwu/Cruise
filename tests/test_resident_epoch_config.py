import pytest

from vllm_ascend_resident_epoch.config import ResidentEpochConfig


def test_fixed_k_graph_environment_requires_a_binary_switch(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_RESIDENT_EPOCH_K6", "1")
    fixed = ResidentEpochConfig.from_env()
    assert fixed.fixed_epoch_graph
    assert fixed.physical_blocks == 4
    assert fixed.blocks_per_request == 1

    monkeypatch.setenv("VLLM_ASCEND_RESIDENT_EPOCH_K6", "0")
    legacy = ResidentEpochConfig.from_env()
    assert not legacy.fixed_epoch_graph
    assert legacy.physical_blocks == 8
    assert legacy.blocks_per_request == 2

    monkeypatch.setenv("VLLM_ASCEND_RESIDENT_EPOCH_K6", "2")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        ResidentEpochConfig.from_env()
