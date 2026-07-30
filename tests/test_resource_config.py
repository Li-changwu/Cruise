import json
from pathlib import Path

import pytest

from prepare_resource_config import prepare_resource_config


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "experiments" / "synthetic-p0" / "numa_config.physical7.json"


def test_prepare_resource_config_maps_device_and_deploy_root(tmp_path) -> None:
    deploy_root = tmp_path / "dataflow-deploy"
    result = prepare_resource_config(TEMPLATE, 0, deploy_root)

    node = result["cluster"][0]["cluster_nodes"][0]
    item = node["item_list"][0]
    assert item == {"item_id": 0, "device_id": 0, "ipaddr": "127.0.0.1"}
    assert node["deploy_res_path"] == str(deploy_root.resolve())
    assert result["cluster"][0]["nodes_topology"]["topos"][0]["devices"] == [0]
    assert json.loads(TEMPLATE.read_text(encoding="utf-8"))["cluster"][0][
        "cluster_nodes"
    ][0]["item_list"][0]["device_id"] == 7


def test_prepare_resource_config_rejects_invalid_physical_npu() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        prepare_resource_config(TEMPLATE, -1, Path("/tmp/deploy"))
