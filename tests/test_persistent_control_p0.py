import json
from pathlib import Path
import sqlite3

from experiments.persistent_control_p0.analyze_p0 import analyze
from experiments.persistent_control_p0.prepare_p0_config import (
    make_config,
    make_deploy_config,
    make_toolchain_config,
)
from experiments.persistent_control_p0.placement_evidence import analyze_placement


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "experiments" / "persistent_control_p0"


def test_p0_function_config_uses_stream_input_and_n_mapping_contract(tmp_path):
    toolchain = tmp_path / "aarch64-target-linux-gnu-g++"
    toolchain.write_text("toolchain", encoding="utf-8")
    toolchain_config = tmp_path / "toolchain.json"
    toolchain_config.write_text(
        json.dumps(make_toolchain_config(toolchain)), encoding="utf-8"
    )
    config = make_config(P0 / "controller", toolchain_config)

    assert config["input_num"] == 1
    assert config["output_num"] == 1
    assert config["heavy_load"] is False
    assert config["func_list"] == [
        {
            "func_name": "persistent_control_p0",
            "inputs_index": [0],
            "outputs_index": [0],
            "stream_input": True,
        }
    ]
    assert config["compiler"] == str(toolchain_config.resolve())
    assert make_toolchain_config(toolchain) == {
        "compiler": [
            {
                "resource_type": "Ascend",
                "toolchain": str(toolchain.resolve()),
            }
        ]
    }
    host = (P0 / "persistent_control_p0_host.cpp").read_text(encoding="utf-8")
    assert "SetContainsNMappingNode(true)" in host
    assert "std::vector<FlowOperator> inputs = {event};" in host
    assert "{node, {0}}" in host
    assert host.index("ge::GEInitialize(config)") < host.index(
        "BuildFlowGraph(function_config.c_str()"
    )
    assert "std::make_shared<ge::Session>(config)" in host
    assert host.index("session->AddGraph(kGraphId, graph)") < host.index(
        "session->CompileGraph(kGraphId)"
    ) < host.index("std::thread drain(FetchOutputs")


def test_p0_deploys_the_heavy_controller_to_the_explicit_device():
    assert make_deploy_config() == {
        "batch_deploy_info": [
            {
                "flow_node_list": ["persistent_control_p0_node"],
                "logic_device_list": "0:0:0:0",
            }
        ]
    }
    host = (P0 / "persistent_control_p0_host.cpp").read_text(encoding="utf-8")
    runner = (P0 / "run_on_910b2.sh").read_text(encoding="utf-8")
    assert '"ge.experiment.data_flow_deploy_info_path", deploy_config.c_str()' in host
    assert "controller_workspace=${scratch}/controller" in runner
    assert 'cp "${controller_source}/CMakeLists.txt"' in runner
    assert '--toolchain-output "${toolchain_config}"' in runner
    assert '--deploy-output "${deploy_config}"' in runner
    assert "analysis_args+=(--mechanism-only)" in runner
    assert "driver_evidence=failure-driver-logs" in runner
    assert '"${evidence}/${driver_evidence}"' in runner
    assert "placement-diagnostic" in runner
    assert 'find "${driver_logs}" -type f -delete' in runner
    assert "placement_probe" in host
    assert '"--mechanism-only"' in (
        P0 / "analyze_p0.py"
    ).read_text(encoding="utf-8")


def test_p0_owner_blocks_when_quiescent_and_polls_only_while_active():
    source = (P0 / "controller" / "persistent_control_p0.cpp").read_text(
        encoding="utf-8"
    )

    assert "const int32_t timeout = active_ ? 0 : -1;" in source
    assert "queues[0]->Dequeue(message, timeout)" in source
    assert "context->SetOutput(0, output)" in source
    assert "RunFlowModel" not in source


def test_p0_host_feeds_only_external_events():
    source = (P0 / "persistent_control_p0_host.cpp").read_text(encoding="utf-8")

    assert source.count("send(kEventAdmit") == 5
    assert source.count("send(kEventCancel") == 1
    assert source.count("send(kEventShutdown") == 2
    assert "send(kEventAdmit, 1, 1, 1, 0, 1000)" in source
    assert '<< " owner_starts=1 feed_calls=" << feed_calls' in source
    assert "FetchOutputs" in source
    assert source.index("std::thread drain(FetchOutputs") < source.index(
        "bool passed = send(kEventAdmit"
    )
    assert "!fetched_output && startup_failures < kFetchStartupRetryLimit" in source


def _write_start(path: Path, owner: int, *, pid: int = 1) -> None:
    lines = [f"P0_OWNER_START pid={pid} owner={owner} quantum_delay_us=1"]
    fetch_calls = 0
    for generation, seed, count in (
        (1, 1000, 1024),
        (2, 2000, 1024),
        (3, 3000, 128),
        (4, 4000, 1024),
    ):
        lines.append(
            f"P0_OUTPUT owner={owner} type=1 event_seq={generation} "
            f"generation={generation} commit_seq=0 state={seed} "
            "remaining=1024 status=0 transaction=1 flags=0"
        )
        fetch_calls += 1
        for commit in range(1, count + 1):
            lines.append(
                f"P0_OUTPUT owner={owner} type=2 event_seq={generation} "
                f"generation={generation} commit_seq={commit} "
                f"state={seed + commit} remaining={1024 - commit} "
                "status=0 transaction=1 flags=0"
            )
            fetch_calls += 1
        retire_type = 4 if generation == 3 else 3
        status = 1 if generation == 3 else 0
        lines.append(
            f"P0_OUTPUT owner={owner} type={retire_type} event_seq={generation} "
            f"generation={generation} commit_seq={count} state={seed + count} "
            f"remaining={1024 - count} status={status} transaction=1 flags=0"
        )
        lines.append(
            f"P0_OUTPUT owner={owner} type=5 event_seq={generation} "
            f"generation={generation} commit_seq={count} state={seed + count} "
            f"remaining={1024 - count} status={generation} transaction=1 flags=0"
        )
        fetch_calls += 2
    lines.append(
        f"P0_OUTPUT owner={owner} type=6 event_seq=6 generation=4 "
        "commit_seq=3200 state=6 remaining=4 status=0 transaction=1 flags=1"
    )
    fetch_calls += 1
    lines.append(
        f"P0_SUMMARY owner={owner} owner_starts=1 feed_calls=6 "
        f"fetch_calls={fetch_calls} admission_events=4 cancel_events=1 "
        "shutdown_events=1 quiescent=4 rejected=0 generation1_commits=1024 "
        "generation2_commits=1024 generation3_commits=128 "
        "generation4_commits=1024 host_cpu_us=1 wall_ms=1 "
        "aicore_tasks_expected=0 compile_status=0 fetch_status=0 "
        "remove_status=0 finalize_status=0"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_profile_status(path: Path, exit_status: int = 0) -> None:
    path.write_text(
        f"profile_exit_status\t{exit_status}\n"
        "aicpu\ton\n"
        "ai_core\ton\n"
        "task_time\tl1\n",
        encoding="utf-8",
    )


def _write_profile_tasks(profile_root: Path, *extra_tasks: tuple[str, str]) -> None:
    task_database = profile_root / "device_0" / "sqlite" / "ascend_task.db"
    task_database.parent.mkdir(parents=True)
    with sqlite3.connect(task_database) as database:
        database.execute(
            "CREATE TABLE AscendTask (device_task_type TEXT, "
            "start_time INTEGER, duration INTEGER, host_task_type TEXT)"
        )
        rows = [
            ("PLACE_HOLDER_SQE", 1, 1, "PROFILING_ENABLE"),
            ("PLACE_HOLDER_SQE", 2, 1, "PROFILING_DISABLE"),
        ]
        rows.extend((device, 10, 5, host) for device, host in extra_tasks)
        database.executemany("INSERT INTO AscendTask VALUES (?, ?, ?, ?)", rows)


def _write_controller_log(root: Path, host_pid: int, expected_outputs: int) -> None:
    path = root / "run" / "device-0" / f"device-{host_pid}_test.log"
    path.parent.mkdir(parents=True)
    prefix = "[INFO] UDF(77,udf_executor):2026-08-08-00:00:00.000 "
    path.write_text(
        prefix
        + "[flow_func_model.cpp:264][Init][tid:77][RUN]: "
        "parse name=persistent_control_p0_pp end, "
        "flowFuncName=persistent_control_p0, "
        "instanceName=persistent_control_p0_pp@0@0_0_0@0.\n"
        + prefix
        + "[flow_func_processor.cpp:321][Init][tid:78][RUN]: "
        "end to init FlowFunc processor, "
        "flow_func_info=persistent_control_p0["
        "persistent_control_p0_pp@0@0_0_0@0].\n"
        + prefix
        + "[flow_func_processor.cpp:903][DumpFlowFuncInfo][tid:77][RUN]: "
        "flow_func_info=persistent_control_p0["
        "persistent_control_p0_pp@0@0_0_0@0], status=8, "
        "statistic info: input ready times=0, call flow func times=2, "
        f"schedule finish times=1, set output times=[{expected_outputs}], "
        "cached nums=[0].\n"
        + prefix
        + "[flow_func_statistic.cpp:124][DumpMetrics][tid:77][RUN]: "
        "model_metrics:name=persistent_control_p0["
        "persistent_control_p0_pp@0@0_0_0@0], min_exec_time=100 us, "
        "max_exec_time=100 us, sub_max_exec_time=0 us, "
        "total_exec_time=100 us, total_exec_num=1.\n"
        + prefix
        + "[main.cpp:432][main][tid:77][RUN]: Flow func executor exit.\n",
        encoding="utf-8",
    )


def _placement_result(tmp_path: Path, *, device: bool = True) -> dict:
    log_root = tmp_path / ("device-placement" if device else "host-placement")
    log_root.mkdir()
    log = log_root / "ge.log"
    if device:
        log.write_text(
            "P0_PLACEMENT_PROBE owner=9001 feed_calls=2 commits=1 quiescent=1 "
            "compile_status=0 fetch_status=0 remove_status=0 finalize_status=0\n"
            "Get pp[persistent_control_p0_pp]'s runnable resource info[Aarch,Ascend] "
            "from node[persistent_control_p0_node]\n"
            "select resource type is [Ascend].\n"
            "Model deployment info, model_name = persistent_control_p0_pp, "
            "model_type = UDF, graph_id = 0, node_type = local, client_id = -1, "
            "device_id = 0.\n"
            "Add model info, model_name = persistent_control_p0_pp, "
            "device_type = 0, model_path = local_context_1/controller.om.\n"
            "[udf_proxy_client.cc:158] LoadProcess:Fork udf process to load model.\n",
            encoding="utf-8",
        )
    else:
        log.write_text(
            "select resource type is [Aarch].\n"
            "Model deployment info, model_name = persistent_control_p0_pp, "
            "model_type = UDF, graph_id = 0, node_type = local, client_id = -1, "
            "device_id = 0.\n"
            "Number [1] of local udfs will untar to local_context_1.\n"
            "The time cost of host udf do untar process is [1] micro seconds.\n"
            "Add model info, model_name = persistent_control_p0_pp, "
            "device_type = 1, model_path = local_context_1/controller.om.\n"
            "[udf_executor_client.cc:321] LoadProcess:Fork udf process to load model.\n",
            encoding="utf-8",
        )
    return analyze_placement([log_root])


def test_p0_placement_gate_accepts_explicit_ascend_device_evidence(tmp_path):
    result = _placement_result(tmp_path)

    assert result["pass"] is True
    assert result["deployment_node_types"] == ["local"]
    assert result["selected_resource_types"] == ["Ascend"]
    assert result["model_device_types"] == [0]
    assert result["device_proxy_records"] == 1
    assert result["placement_probe_records"] == 1
    assert not any(result["forbidden_evidence_counts"].values())


def test_p0_placement_gate_rejects_host_local_udf_evidence(tmp_path):
    result = _placement_result(tmp_path, device=False)

    assert result["pass"] is False
    assert result["deployment_node_types"] == ["local"]
    assert result["forbidden_evidence_counts"]["host_udf_untar"] == 1
    assert result["forbidden_evidence_counts"]["host_udf_executor"] == 1


def test_p0_analyzer_accepts_three_exact_starts(tmp_path):
    logs = []
    for index in range(3):
        path = tmp_path / f"start-{index}.log"
        _write_start(path, 1000 + index)
        logs.append(path)
    before = tmp_path / "before.txt"
    before.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")
    after = []
    for index, usage in enumerate((51, 52, 51)):
        path = tmp_path / f"after-{index}.txt"
        path.write_text(f"HBM Usage Rate(%) : {usage}\n", encoding="utf-8")
        after.append(path)
    profile_status = tmp_path / "profile-status.tsv"
    _write_profile_status(profile_status)
    profile_root = tmp_path / "profile"
    _write_profile_tasks(profile_root)
    profile_log = tmp_path / "profile.log"
    _write_start(profile_log, 2001, pid=99)
    controller_logs = tmp_path / "driver-logs"
    profile_outputs = sum(
        1
        for line in profile_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("P0_OUTPUT ")
    )
    _write_controller_log(controller_logs, 99, profile_outputs)
    after_profile = tmp_path / "after-profile.txt"
    after_profile.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")

    result = analyze(
        logs,
        before,
        after,
        profile_status,
        profile_root,
        _placement_result(tmp_path),
        profile_log=profile_log,
        controller_log_roots=[controller_logs],
        npu_after_profile=after_profile,
    )

    assert result["pass"] is True
    assert result["mechanism_pass"] is True
    assert result["profile_pass"] is True
    assert result["profile"]["ai_core_task_count"] == 0
    assert result["profile"]["ai_cpu_task_count"] == 0
    assert result["profile"]["task_evidence_available"] is True
    assert result["profile"]["capture_started"] is True
    assert result["profile"]["capture_stopped"] is True
    assert result["profile"]["controller"]["pass"] is True
    assert result["profile"]["controller"]["device_executor_pids"] == [77]
    assert (
        result["profile"]["aicpu_task_visibility"]
        == "flowfunc_not_exported_as_ordinary_task"
    )
    assert json.loads(json.dumps(result))["owner_instances_unique"] is True


def test_p0_analyzer_rejects_an_ai_core_task(tmp_path):
    logs = []
    for index in range(3):
        path = tmp_path / f"start-{index}.log"
        _write_start(path, 1000 + index)
        logs.append(path)
    before = tmp_path / "before.txt"
    before.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")
    after = []
    for index in range(3):
        path = tmp_path / f"after-{index}.txt"
        path.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")
        after.append(path)
    profile_status = tmp_path / "profile-status.tsv"
    _write_profile_status(profile_status)
    profile_root = tmp_path / "profile"
    _write_profile_tasks(profile_root, ("AI_CORE", "unexpected_kernel"))
    profile_log = tmp_path / "profile.log"
    _write_start(profile_log, 2001, pid=99)
    controller_logs = tmp_path / "driver-logs"
    profile_outputs = sum(
        1
        for line in profile_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("P0_OUTPUT ")
    )
    _write_controller_log(controller_logs, 99, profile_outputs)
    after_profile = tmp_path / "after-profile.txt"
    after_profile.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")

    result = analyze(
        logs,
        before,
        after,
        profile_status,
        profile_root,
        _placement_result(tmp_path),
        profile_log=profile_log,
        controller_log_roots=[controller_logs],
        npu_after_profile=after_profile,
    )

    assert result["mechanism_pass"] is True
    assert result["profile_pass"] is False
    assert result["pass"] is False
    assert result["profile"]["ai_cpu_task_count"] == 0
    assert result["profile"]["ai_core_task_count"] == 1


def test_p0_analyzer_rejects_generic_ai_cpu_without_controller_attribution(
    tmp_path,
):
    logs = []
    for index in range(3):
        path = tmp_path / f"start-{index}.log"
        _write_start(path, 1000 + index)
        logs.append(path)
    before = tmp_path / "before.txt"
    before.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")
    after = []
    for index in range(3):
        path = tmp_path / f"after-{index}.txt"
        path.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")
        after.append(path)
    profile_status = tmp_path / "profile-status.tsv"
    _write_profile_status(profile_status)
    profile_root = tmp_path / "profile"
    _write_profile_tasks(profile_root, ("AI_CPU", "unattributed_infrastructure"))
    profile_log = tmp_path / "profile.log"
    _write_start(profile_log, 2001, pid=99)
    controller_logs = tmp_path / "driver-logs"
    controller_logs.mkdir()
    after_profile = tmp_path / "after-profile.txt"
    after_profile.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")

    result = analyze(
        logs,
        before,
        after,
        profile_status,
        profile_root,
        _placement_result(tmp_path),
        profile_log=profile_log,
        controller_log_roots=[controller_logs],
        npu_after_profile=after_profile,
    )

    assert result["mechanism_pass"] is True
    assert result["profile_pass"] is False
    assert result["profile"]["ai_cpu_task_count"] == 1
    assert result["profile"]["controller"]["pass"] is False


def test_p0_mechanism_gate_rejects_host_placement(tmp_path):
    logs = []
    for index in range(3):
        path = tmp_path / f"start-{index}.log"
        _write_start(path, 1000 + index)
        logs.append(path)
    before = tmp_path / "before.txt"
    before.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")
    after = []
    for index in range(3):
        path = tmp_path / f"after-{index}.txt"
        path.write_text("HBM Usage Rate(%) : 51\n", encoding="utf-8")
        after.append(path)
    profile_status = tmp_path / "profile-status.tsv"
    _write_profile_status(profile_status, 96)
    profile_root = tmp_path / "profile"
    profile_root.mkdir()

    result = analyze(
        logs,
        before,
        after,
        profile_status,
        profile_root,
        _placement_result(tmp_path, device=False),
    )

    assert all(start["pass"] for start in result["starts"])
    assert result["mechanism_pass"] is False
    assert result["placement"]["pass"] is False
