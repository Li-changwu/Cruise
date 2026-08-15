import json
from pathlib import Path

from experiments.persistent_recurrence_p1.analyze_p1 import (
    EXPECTED_OUTPUTS,
    analyze,
    expected_outputs,
    verify_start,
)
from experiments.persistent_recurrence_p1.placement_evidence import analyze_placement
from experiments.persistent_recurrence_p1.prepare_p1_config import (
    make_deploy_config,
    make_function_config,
    make_graph_config,
    make_toolchain_config,
)


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "experiments" / "persistent_recurrence_p1"


def test_p1_configs_bind_device_controller_and_fixed_add_graph(tmp_path):
    toolchain = tmp_path / "aarch64-target-linux-gnu-g++"
    toolchain.write_text("toolchain", encoding="utf-8")
    toolchain_config = tmp_path / "toolchain.json"
    toolchain_config.write_text(
        json.dumps(make_toolchain_config(toolchain)), encoding="utf-8"
    )

    function = make_function_config(P1 / "controller", toolchain_config)
    assert function["heavy_load"] is False
    assert function["func_list"] == [
        {
            "func_name": "persistent_recurrence_p1",
            "inputs_index": [0],
            "outputs_index": [0],
            "stream_input": True,
        }
    ]
    assert function["target_bin"] == "libpersistent_recurrence_p1.so"
    assert make_toolchain_config(toolchain)["compiler"][0]["resource_type"] == (
        "Ascend"
    )
    assert make_graph_config()["inputs_tensor_desc"] == [
        {"data_type": "DT_INT32", "shape": [4]},
        {"data_type": "DT_INT32", "shape": [4]},
    ]
    assert make_deploy_config() == {
        "batch_deploy_info": [
            {
                "flow_node_list": ["persistent_recurrence_p1_node"],
                "logic_device_list": "0:0:0:0",
            }
        ]
    }


def test_p1_controller_invokes_aicore_only_for_credit_eligible_rows():
    source = (P1 / "controller" / "persistent_recurrence_p1.cpp").read_text(
        encoding="utf-8"
    )

    assert 'RunFlowModel(\n        "recurrence_graph"' in source
    assert "active_[row] && credits_[row] > 0" in source
    assert "delta_values[row] = eligible[row] ? 1 : 0;" in source
    assert "--credits_[row];" in source
    assert "kSyntheticCommitPaceUs = 1000" in source
    assert "eligible_rows) * kSyntheticCommitPaceUs" in source
    assert source.index("RunFlowModel(") < source.index(
        "kOutputCommit, cohort_event_seq_"
    )
    assert "const int32_t timeout = HasEligibleRow() ? 0 : -1;" in source
    assert "FetchDataFlowGraph" not in source


def test_p1_host_exposes_only_admit_credit_and_shutdown_events():
    source = (P1 / "persistent_recurrence_p1_host.cpp").read_text(
        encoding="utf-8"
    )

    assert 'AddInvokedClosure("recurrence_graph", recurrence)' in source
    assert 'ge::op::Add("persistent_recurrence_p1_add")' in source
    assert "SetContainsNMappingNode(true)" in source
    assert source.count("send(kEventCredit") == 1
    assert source.count("send(kEventShutdown") == 2
    assert "kExpectedAicoreCalls = 752" in source
    assert "kExpectedCommits = 1280" in source
    assert '"P1_CREDIT_ISOLATION owner="' in source
    assert '"P1_PRE_SHUTDOWN owner="' in source
    assert "RunGraph" not in source


def test_p1_protocol_freezes_exact_credit_isolation_gate():
    protocol = (P1 / "protocol.md").read_text(encoding="utf-8")

    assert "`[16, 256, 256, 256]`" in protocol
    assert "exactly 752 times" in protocol
    assert "All 1,280 commits must be visible before shutdown" in protocol
    assert "Host feed count is exactly four" in protocol


def test_p1_runner_bounds_and_reviews_intermediate_evidence():
    runner = (P1 / "run_on_910b2.sh").read_text(encoding="utf-8")

    assert "--storage-limit=200MB" in runner
    assert "STORAGE_GUARD_MAX_SCRATCH_GIB=8" in runner
    assert "elewise_calculation_ops.h" in runner
    assert "review_intermediates after-placement" in runner
    assert 'review_intermediates "after-start-${start}"' in runner
    assert "review_intermediates after-profile" in runner
    assert 'find "${driver_logs}" -type f -delete' in runner
    assert "failure-driver-logs" in runner
    assert "analysis_args+=(--mechanism-only)" in runner
    assert "placement-diagnostic" in runner


def _write_start(
    path: Path,
    owner: int,
    *,
    pid: int = 1,
    corrupt_state: bool = False,
    blocked_commits: int = 16,
) -> None:
    outputs = expected_outputs(owner)
    if corrupt_state:
        target = next(
            output
            for output in outputs
            if output["type"] == 2
            and output["cohort"] == 2
            and output["row"] == 2
            and output["commit_seq"] == 23
        )
        target["state"] += 1
    lines = [f"P1_OWNER_START pid={pid} owner={owner} mode=0"]
    lines.extend(
        "P1_OUTPUT "
        + " ".join(f"{key}={value}" for key, value in output.items())
        for output in outputs
    )
    lines.append(
        f"P1_CREDIT_ISOLATION owner={owner} cohort=2 blocked_row=0 "
        f"blocked_commits={blocked_commits} peer1_commits=256 "
        "peer2_commits=256 peer3_commits=256"
    )
    lines.append(
        f"P1_PRE_SHUTDOWN owner={owner} commits=1280 fetch_calls=1290"
    )
    lines.append(
        f"P1_SUMMARY owner={owner} owner_starts=1 feed_calls=4 "
        f"fetch_calls={EXPECTED_OUTPUTS} admission_events=2 credit_events=1 "
        "shutdown_events=1 quiescent=2 rejected=0 b1_row0_commits=256 "
        "b4_row0_commits=256 b4_row1_commits=256 b4_row2_commits=256 "
        "b4_row3_commits=256 blocked_row0_commits=16 aicore_calls=752 "
        "total_commits=1280 pre_shutdown_commits=1280 "
        "pre_shutdown_fetches=1290 host_cpu_us=100 wall_ms=100 "
        "compile_status=0 fetch_status=0 remove_status=0 finalize_status=0"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_profile_status(path: Path) -> None:
    path.write_text(
        "profile_exit_status\t0\n"
        "aicpu\ton\n"
        "ai_core\ton\n"
        "task_time\tl1\n",
        encoding="utf-8",
    )


def _write_profile_tasks(
    profile_root: Path,
    *,
    task_count: int = 752,
    kernel_name: str = "persistent_recurrence_p1_add",
) -> None:
    path = profile_root / "mindstudio_profiler_output" / "task_time_test.csv"
    path.parent.mkdir(parents=True)
    lines = ["Device_id,kernel_name,kernel_type"]
    lines.extend(
        f"0,{kernel_name},AI_VECTOR_CORE" for _ in range(task_count)
    )
    lines.extend(
        (
            "0,N/A,PROFILING_ENABLE",
            "0,N/A,PROFILING_DISABLE",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_profile_op_summary(
    profile_root: Path,
    *,
    task_count: int = 752,
    op_name: str = (
        "persistent_recurrence_p1_graph_pp/"
        "persistent_recurrence_p1_add"
    ),
) -> None:
    path = profile_root / "mindstudio_profiler_output" / "op_summary_test.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Device_id,Op Name,OP Type,Task Type"]
    lines.extend(
        f"0,{op_name},Add,AI_VECTOR_CORE" for _ in range(task_count)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_controller_log(root: Path, host_pid: int) -> None:
    path = root / "run" / "device-0" / f"device-{host_pid}_test.log"
    path.parent.mkdir(parents=True)
    prefix = "[INFO] UDF(77,udf_executor):2026-08-08-00:00:00.000 "
    path.write_text(
        prefix
        + "parse name=persistent_recurrence_p1_pp end, "
        "flowFuncName=persistent_recurrence_p1, "
        "instanceName=persistent_recurrence_p1_pp@0@0_0_0@0.\n"
        + prefix
        + "end to init FlowFunc processor, "
        "flow_func_info=persistent_recurrence_p1["
        "persistent_recurrence_p1_pp@0@0_0_0@0].\n"
        + prefix
        + "flow_func_info=persistent_recurrence_p1["
        "persistent_recurrence_p1_pp@0@0_0_0@0], status=8, "
        "call flow func times=2, schedule finish times=1, "
        "set output times=[1291].\n"
        + prefix
        + "model_metrics:name=persistent_recurrence_p1["
        "persistent_recurrence_p1_pp@0@0_0_0@0], min_exec_time=100 us, "
        "max_exec_time=100 us, sub_max_exec_time=0 us, "
        "total_exec_time=100 us, total_exec_num=1.\n"
        + prefix
        + "Flow func executor exit.\n",
        encoding="utf-8",
    )


def _placement_result(tmp_path: Path, *, device: bool = True) -> dict:
    log_root = tmp_path / ("device-placement" if device else "host-placement")
    log_root.mkdir()
    log = log_root / "ge.log"
    if device:
        log.write_text(
            "P1_PLACEMENT_PROBE owner=9001 feed_calls=2 commits=256 "
            "aicore_calls=256 quiescent=1 compile_status=0 fetch_status=0 "
            "remove_status=0 finalize_status=0\n"
            "Get pp[persistent_recurrence_p1_pp]'s runnable resource "
            "info[Aarch,Ascend] from node[persistent_recurrence_p1_node]\n"
            "select resource type is [Ascend].\n"
            "Model deployment info, model_name = persistent_recurrence_p1_pp, "
            "model_type = UDF, graph_id = 0, node_type = local, client_id = -1, "
            "device_id = 0.\n"
            "Add model info, model_name = persistent_recurrence_p1_pp, "
            "device_type = 0, model_path = local_context_1/controller.om.\n"
            "[udf_proxy_client.cc:158] LoadProcess:Fork udf process.\n",
            encoding="utf-8",
        )
    else:
        log.write_text(
            "select resource type is [Aarch].\n"
            "Model deployment info, model_name = persistent_recurrence_p1_pp, "
            "node_type = local, device_id = 0.\n"
            "Number [1] of local udfs will untar to local_context_1.\n"
            "The time cost of host udf do untar process is [1] micro seconds.\n"
            "Add model info, model_name = persistent_recurrence_p1_pp, "
            "device_type = 1, model_path = local_context_1/controller.om.\n"
            "[udf_executor_client.cc:321] LoadProcess:Fork udf process.\n",
            encoding="utf-8",
        )
    return analyze_placement([log_root])


def _analyze_fixture(
    tmp_path: Path,
    *,
    profile_task_count: int = 752,
    kernel_name: str = "persistent_recurrence_p1_add",
    op_summary_name: str | None = None,
    placement_device: bool = True,
) -> dict:
    logs = []
    for index in range(3):
        path = tmp_path / f"start-{index}.log"
        _write_start(path, 1001 + index)
        logs.append(path)
    before = tmp_path / "before.txt"
    before.write_text("HBM Usage Rate(%) : 58\n", encoding="utf-8")
    after = []
    for index, usage in enumerate((58, 59, 58)):
        path = tmp_path / f"after-{index}.txt"
        path.write_text(f"HBM Usage Rate(%) : {usage}\n", encoding="utf-8")
        after.append(path)
    profile_status = tmp_path / "profile-status.tsv"
    _write_profile_status(profile_status)
    profile_root = tmp_path / "profile"
    _write_profile_tasks(
        profile_root, task_count=profile_task_count, kernel_name=kernel_name
    )
    if op_summary_name is not None:
        _write_profile_op_summary(
            profile_root,
            task_count=profile_task_count,
            op_name=op_summary_name,
        )
    profile_log = tmp_path / "profile.log"
    _write_start(profile_log, 2001, pid=99)
    controller_logs = tmp_path / "driver-logs"
    _write_controller_log(controller_logs, 99)
    after_profile = tmp_path / "after-profile.txt"
    after_profile.write_text("HBM Usage Rate(%) : 58\n", encoding="utf-8")
    return analyze(
        logs,
        before,
        after,
        profile_status,
        profile_root,
        _placement_result(tmp_path, device=placement_device),
        profile_log=profile_log,
        controller_log_roots=[controller_logs],
        npu_after_profile=after_profile,
    )


def test_p1_placement_gate_accepts_device_and_rejects_host(tmp_path):
    accepted = _placement_result(tmp_path)
    rejected = _placement_result(tmp_path, device=False)

    assert accepted["pass"] is True
    assert accepted["selected_resource_types"] == ["Ascend"]
    assert accepted["model_device_types"] == [0]
    assert rejected["pass"] is False
    assert rejected["forbidden_evidence_counts"]["host_udf_untar"] == 1
    assert rejected["forbidden_evidence_counts"]["host_udf_executor"] == 1


def test_p1_start_verifier_accepts_exact_sequence(tmp_path):
    path = tmp_path / "start.log"
    _write_start(path, 1001)

    result = verify_start(path)

    assert result["pass"] is True
    assert result["outputs"] == 1291
    assert result["summary"]["aicore_calls"] == 752


def test_p1_start_verifier_rejects_state_and_credit_isolation(tmp_path):
    state_path = tmp_path / "bad-state.log"
    credit_path = tmp_path / "bad-credit.log"
    _write_start(state_path, 1001, corrupt_state=True)
    _write_start(credit_path, 1002, blocked_commits=17)

    state_result = verify_start(state_path)
    credit_result = verify_start(credit_path)

    assert state_result["pass"] is False
    assert any("output[" in error for error in state_result["errors"])
    assert credit_result["pass"] is False
    assert any("credit isolation=" in error for error in credit_result["errors"])


def test_p1_analyzer_accepts_exact_mechanism_and_target_add_profile(tmp_path):
    result = _analyze_fixture(tmp_path)

    assert result["pass"] is True
    assert result["mechanism_pass"] is True
    assert result["profile_pass"] is True
    assert result["profile"]["ai_core_task_count"] == 752
    assert result["profile"]["target_add_task_count"] == 752
    assert result["profile"]["controller"]["device_executor_pids"] == [77]
    assert result["owner_instances_unique"] is True


def test_p1_analyzer_attributes_hashed_kernel_from_device_op_summary(tmp_path):
    result = _analyze_fixture(
        tmp_path,
        kernel_name=(
            "te_add_4dca83424119a529552268166809214c7781d23394f3a2e5e"
            "bdcdb827c1a085f__kernel0"
        ),
        op_summary_name=(
            "persistent_recurrence_p1_graph_pp/"
            "persistent_recurrence_p1_add"
        ),
    )

    assert result["profile_pass"] is True
    assert result["profile"]["target_add_task_count"] == 752


def test_p1_analyzer_rejects_unrelated_add_op_summary(tmp_path):
    result = _analyze_fixture(
        tmp_path,
        kernel_name=(
            "te_add_4dca83424119a529552268166809214c7781d23394f3a2e5e"
            "bdcdb827c1a085f__kernel0"
        ),
        op_summary_name="unrelated_graph/unrelated_add",
    )

    assert result["profile_pass"] is False
    assert result["profile"]["target_add_task_count"] == 0


def test_p1_analyzer_rejects_wrong_aicore_count(tmp_path):
    result = _analyze_fixture(tmp_path, profile_task_count=751)

    assert result["mechanism_pass"] is True
    assert result["profile_pass"] is False
    assert result["profile"]["ai_core_task_count"] == 751


def test_p1_analyzer_rejects_unattributed_aicore_tasks(tmp_path):
    result = _analyze_fixture(tmp_path, kernel_name="MatMul_unrelated")

    assert result["mechanism_pass"] is True
    assert result["profile_pass"] is False
    assert result["profile"]["ai_core_task_count"] == 752
    assert result["profile"]["target_add_task_count"] == 0


def test_p1_mechanism_gate_rejects_host_placement(tmp_path):
    result = _analyze_fixture(tmp_path, placement_device=False)

    assert all(start["pass"] for start in result["starts"])
    assert result["mechanism_pass"] is False
    assert result["placement"]["pass"] is False
