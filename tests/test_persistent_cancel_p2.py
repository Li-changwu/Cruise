import json
from pathlib import Path

from experiments.persistent_cancel_p2.analyze_p2 import analyze, verify_start
from experiments.persistent_cancel_p2.placement_evidence import analyze_placement
from experiments.persistent_cancel_p2.prepare_p2_config import (
    make_deploy_config,
    make_function_config,
    make_graph_config,
    make_toolchain_config,
)


ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "experiments" / "persistent_cancel_p2"


def test_p2_config_and_device_control_contract(tmp_path):
    toolchain = tmp_path / "aarch64-target-linux-gnu-g++"
    toolchain.write_text("toolchain", encoding="utf-8")
    toolchain_config = tmp_path / "toolchain.json"
    toolchain_config.write_text(
        json.dumps(make_toolchain_config(toolchain)), encoding="utf-8"
    )
    config = make_function_config(P2 / "controller", toolchain_config)

    assert config["target_bin"] == "libpersistent_cancel_p2.so"
    assert config["heavy_load"] is False
    assert config["func_list"] == [
        {
            "func_name": "persistent_cancel_p2",
            "inputs_index": [0],
            "outputs_index": [0],
            "stream_input": True,
        }
    ]
    assert make_graph_config()["inputs_tensor_desc"] == [
        {"data_type": "DT_INT32", "shape": [4]},
        {"data_type": "DT_INT32", "shape": [4]},
    ]
    assert make_deploy_config() == {
        "batch_deploy_info": [
            {
                "flow_node_list": ["persistent_cancel_p2_node"],
                "logic_device_list": "0:0:0:0",
            }
        ]
    }

    controller = (P2 / "controller" / "persistent_cancel_p2.cpp").read_text(
        encoding="utf-8"
    )
    host = (P2 / "persistent_cancel_p2_host.cpp").read_text(encoding="utf-8")
    runner = (P2 / "run_on_910b2.sh").read_text(encoding="utf-8")
    assert "const int32_t timeout = HasEligibleRow() ? 0 : -1;" in controller
    assert controller.index("HandleEvent(context, event, shutdown)") < controller.index(
        "RunQuantum(context)"
    )
    assert "context->RunFlowModel(\"recurrence_graph\"" in controller
    assert "active_[row] = false;" in controller
    cancel_handler = controller[controller.index("int32_t HandleCancel") :]
    assert cancel_handler.index("active_[row] = false;") < cancel_handler.index(
        "EmitRow(context, kOutputRetireCancelled"
    )
    assert "SameEvent(event, last_event_)" in controller
    assert "kOutputCumulativeAck" in controller
    assert "int64_t row = -1;" in host
    assert "kStressCohorts = 250" in host
    assert "kStressCancellations = kStressCohorts * kRows" in host
    assert "FeedEvent(session, event, ++transport_seq)" in host
    assert "flow_info.SetTransactionId(transport_seq)" in host
    assert "flow_info.SetTransactionId(static_cast<uint64_t>(event.seq))" not in host
    assert "FetchOutputs" in host
    assert "RunFlowGraph" not in host
    assert "STORAGE_GUARD_MAX_SCRATCH_GIB=8" in runner
    assert 'run_mode}" == single-start-diagnostic' in runner
    assert "PERSISTENT_CANCEL_P2_SINGLE_START_COMPLETE" in runner
    assert "--storage-limit=200MB" in runner
    assert 'review_intermediates "after-start-${start}"' in runner


class _OutputWriter:
    def __init__(self, owner: int):
        self.owner = owner
        self.lines: list[str] = []
        self.aicore_calls = 0
        self.total_commits = 0
        self.total_retired = 0
        self.quiescent = 0

    def emit(
        self,
        output_type: int,
        event_seq: int,
        request: int,
        generation: int,
        row: int,
        commit: int,
        state: int,
        remaining: int,
        credit: int,
        cancel_seq: int = 0,
        status: int = 0,
        latency_us: int = -1,
        flags: int = 0,
    ) -> None:
        self.lines.append(
            f"P2_OUTPUT owner={self.owner} type={output_type} "
            f"event_seq={event_seq} request={request} generation={generation} "
            f"row={row} commit_seq={commit} state={state} "
            f"remaining={remaining} credit={credit} cancel_seq={cancel_seq} "
            f"status={status} aicore_calls={self.aicore_calls} "
            f"total_commits={self.total_commits} "
            f"total_retired={self.total_retired} quiescent={self.quiescent} "
            f"host_cancel_latency_us={latency_us} transaction={event_seq} "
            f"flags={flags}"
        )

    def admit(
        self,
        event_seq: int,
        request: int,
        generation: int,
        row: int,
        seed: int,
        target: int,
        credit: int,
    ) -> None:
        self.emit(1, event_seq, request, generation, row, 0, seed, target, credit)

    def commits(
        self,
        event_seq: int,
        request: int,
        generation: int,
        row: int,
        seed: int,
        target: int,
        first: int,
        last: int,
        initial_credit: int,
        credit_grant: int = 0,
        credit_boundary: int = 0,
    ) -> None:
        for commit in range(first, last + 1):
            self.aicore_calls += 1
            self.total_commits += 1
            credit = initial_credit - commit
            if commit > credit_boundary:
                credit += credit_grant
            self.emit(
                2,
                event_seq,
                request,
                generation,
                row,
                commit,
                seed + commit,
                target - commit,
                credit,
            )

    def retire(
        self,
        output_type: int,
        event_seq: int,
        request: int,
        generation: int,
        row: int,
        seed: int,
        target: int,
        commit: int,
        credit: int,
        latency_us: int = -1,
    ) -> None:
        self.total_retired += 1
        self.emit(
            output_type,
            event_seq,
            request,
            generation,
            row,
            commit,
            seed + commit,
            target - commit,
            credit,
            cancel_seq=1 if output_type == 4 else 0,
            latency_us=latency_us,
        )

    def quiet(
        self,
        event_seq: int,
        request: int,
        generation: int,
        row: int,
        seed: int,
        target: int,
        commit: int,
        credit: int,
    ) -> None:
        self.quiescent += 1
        self.emit(
            6,
            event_seq,
            request,
            generation,
            row,
            commit,
            seed + commit,
            target - commit,
            credit,
        )


def _write_start(path: Path, owner: int, *, pid: int = 1) -> int:
    output = _OutputWriter(owner)
    output.lines.append(f"P2_OWNER_START pid={pid} owner={owner} mode=0")

    output.admit(1, 101, 1, 0, 1000, 8, 0)
    output.emit(7, 1, 101, 1, 0, 0, 1000, 8, 0, status=12)
    output.retire(4, 2, 101, 1, 0, 1000, 8, 0, 0)
    output.quiet(2, 101, 1, 0, 1000, 8, 0, 0)
    output.emit(7, 2, 101, 1, 0, 0, 1000, 8, 0, 1, 12)

    output.admit(3, 102, 2, 0, 2000, 4, 4)
    output.commits(3, 102, 2, 0, 2000, 4, 1, 4, 4)
    output.retire(3, 3, 102, 2, 0, 2000, 4, 4, 0)
    output.quiet(3, 102, 2, 0, 2000, 4, 4, 0)
    output.emit(8, 4, 102, 2, 0, 4, 2004, 0, 0, 1, 9)
    output.emit(8, 3, 102, 2, -1, 0, 2000, 4, 4, status=2)
    output.emit(8, 5, 102, 2, 0, 4, 2004, 0, 0, 2, 1)

    output.admit(6, 103, 3, 0, 3000, 256, 256)
    output.commits(6, 103, 3, 0, 3000, 256, 1, 32, 256)
    output.retire(4, 7, 103, 3, 0, 3000, 256, 32, 224)
    output.quiet(7, 103, 3, 0, 3000, 256, 32, 224)
    output.emit(7, 7, 103, 3, 0, 32, 3032, 224, 224, 1, 12)

    output.admit(8, 104, 4, 0, 4000, 64, 16)
    output.emit(8, 9, 104, 4, 0, 0, 4000, 64, 16, 2, 4)
    output.commits(9, 104, 4, 0, 4000, 64, 1, 16, 16)
    output.emit(5, 10, 104, 4, 0, 16, 4016, 48, 48)
    output.emit(7, 10, 104, 4, 0, 16, 4016, 48, 48, status=12)
    output.commits(10, 104, 4, 0, 4000, 64, 17, 20, 16, 48, 16)
    output.retire(4, 11, 104, 4, 0, 4000, 64, 20, 44)
    output.quiet(11, 104, 4, 0, 4000, 64, 20, 44)

    capacity = [
        (201, 10, 0, 5000),
        (202, 11, 1, 5100),
        (203, 12, 2, 5200),
        (204, 13, 3, 5300),
    ]
    for offset, (request, generation, row, seed) in enumerate(capacity):
        output.admit(12 + offset, request, generation, row, seed, 256, 256)
    for request, generation, row, seed in capacity:
        output.commits(15, request, generation, row, seed, 256, 1, 1, 256)
    output.emit(8, 16, 205, 14, -1, 0, 6000, 256, 256, status=3)
    output.retire(4, 17, 201, 10, 0, 5000, 256, 1, 255)
    output.admit(18, 205, 14, 0, 6000, 256, 256)
    output.commits(18, 205, 14, 0, 6000, 256, 1, 1, 256)
    for event_seq, (request, generation, row, seed) in zip(
        range(19, 23),
        [(205, 14, 0, 6000), capacity[1], capacity[2], capacity[3]],
    ):
        output.retire(4, event_seq, request, generation, row, seed, 256, 1, 255)
        if event_seq == 22:
            output.quiet(event_seq, request, generation, row, seed, 256, 1, 255)

    stress_latencies = []
    event_seq = 23
    for cohort in range(250):
        keys = []
        for row in range(4):
            index = cohort * 4 + row
            request = 10000 + index
            generation = 1000 + index
            seed = 100000 + generation
            keys.append((request, generation, row, seed))
            output.admit(event_seq, request, generation, row, seed, 256, 256)
            event_seq += 1
        for request, generation, row, seed in keys:
            output.commits(event_seq - 1, request, generation, row, seed, 256, 1, 1, 256)
        for request, generation, row, seed in keys:
            latency = 1000 + ((request - 10000) % 100) * 100
            stress_latencies.append(latency)
            output.retire(
                4,
                event_seq,
                request,
                generation,
                row,
                seed,
                256,
                1,
                255,
                latency,
            )
            if row == 3:
                output.quiet(
                    event_seq, request, generation, row, seed, 256, 1, 255
                )
            event_seq += 1

    p50 = sorted(stress_latencies)[499]
    p95 = sorted(stress_latencies)[949]
    p99 = sorted(stress_latencies)[989]
    output.emit(
        9,
        event_seq,
        0,
        0,
        -1,
        output.aicore_calls,
        output.total_commits,
        0,
        0,
        output.total_retired,
        flags=1,
    )
    fetch_calls = len([line for line in output.lines if line.startswith("P2_OUTPUT ")])
    output.lines.append(
        f"P2_MATRIX owner={owner} before_first_commits=0 during_commits=32 "
        "completed_commits=4 blocked_commits=16 reused_row=0 "
        "duplicate_acks=4 rejected=5 matrix_cancelled=8 "
        "retirement_before_reuse=1 stale_generation_isolated=1"
    )
    output.lines.append(
        f"P2_CANCEL_LATENCY owner={owner} samples=1000 p50_us={p50} "
        f"p95_us={p95} p99_us={p99} max_us={max(stress_latencies)}"
    )
    output.lines.append(
        f"P2_SUMMARY owner={owner} owner_starts=1 feed_calls=2028 "
        f"fetch_calls={fetch_calls} matrix_admissions=9 matrix_cancelled=8 "
        "stress_admissions=1000 stress_cancelled=1000 duplicate_acks=4 "
        f"rejected=5 quiescent=255 aicore_calls={output.aicore_calls} "
        f"total_commits={output.total_commits} total_retired=1009 "
        f"latency_samples=1000 latency_p95_us={p95} latency_p99_us={p99} "
        "host_cpu_us=1 wall_ms=1 compile_status=0 fetch_status=0 "
        "remove_status=0 finalize_status=0"
    )
    path.write_text("\n".join(output.lines) + "\n", encoding="utf-8")
    return output.aicore_calls


def _write_profile_status(path: Path) -> None:
    path.write_text(
        "profile_exit_status\t0\n"
        "aicpu\ton\n"
        "ai_core\ton\n"
        "task_time\tl1\n",
        encoding="utf-8",
    )


def _write_profile_tasks(profile_root: Path, count: int, *, target: bool = True) -> None:
    output = profile_root / "mindstudio_profiler_output"
    output.mkdir(parents=True)
    task_time = output / "task_time_test.csv"
    task_time.write_text(
        "Device_id,kernel_name,kernel_type\n"
        + "\n".join(
            ["0,N/A,PROFILING_ENABLE"]
            + [f"0,te_add_hash__kernel0,AI_VECTOR_CORE" for _ in range(count)]
            + ["0,N/A,PROFILING_DISABLE"]
        )
        + "\n",
        encoding="utf-8",
    )
    op_name = (
        "persistent_cancel_p2_graph_pp/persistent_cancel_p2_add"
        if target
        else "unrelated_graph/unrelated_add"
    )
    (output / "op_summary_test.csv").write_text(
        "Device_id,Op Name,OP Type,Task Type\n"
        + "\n".join(
            f"0,{op_name},Add,AI_VECTOR_CORE" for _ in range(count)
        )
        + "\n",
        encoding="utf-8",
    )


def _write_controller_log(root: Path, host_pid: int, expected_outputs: int) -> None:
    path = root / "run" / "device-0" / f"device-{host_pid}_test.log"
    path.parent.mkdir(parents=True)
    prefix = "[INFO] UDF(77,udf_executor):2026-08-09-00:00:00.000 "
    path.write_text(
        prefix
        + "parse name=persistent_cancel_p2_pp end, "
        "flowFuncName=persistent_cancel_p2, "
        "instanceName=persistent_cancel_p2_pp@0@0_0_0@0.\n"
        + prefix
        + "end to init FlowFunc processor, "
        "flow_func_info=persistent_cancel_p2["
        "persistent_cancel_p2_pp@0@0_0_0@0].\n"
        + prefix
        + "flow_func_info=persistent_cancel_p2["
        "persistent_cancel_p2_pp@0@0_0_0@0], status=8, "
        "call flow func times=2, schedule finish times=1, "
        f"set output times=[{expected_outputs}].\n"
        + prefix
        + "model_metrics:name=persistent_cancel_p2["
        "persistent_cancel_p2_pp@0@0_0_0@0], min_exec_time=100 us, "
        "max_exec_time=100 us, sub_max_exec_time=0 us, "
        "total_exec_time=100 us, total_exec_num=1.\n"
        + prefix
        + "Flow func executor exit.\n",
        encoding="utf-8",
    )


def _placement_result(tmp_path: Path, *, device: bool = True) -> dict:
    root = tmp_path / ("device-placement" if device else "host-placement")
    root.mkdir()
    log = root / "ge.log"
    if device:
        log.write_text(
            "P2_PLACEMENT_PROBE owner=9001 feed_calls=2 commits=1 "
            "aicore_calls=1 retired=1 quiescent=1 compile_status=0 "
            "fetch_status=0 remove_status=0 finalize_status=0\n"
            "Get pp[persistent_cancel_p2_pp]'s runnable resource "
            "info[Aarch,Ascend] from node[persistent_cancel_p2_node]\n"
            "select resource type is [Ascend].\n"
            "Model deployment info, model_name = persistent_cancel_p2_pp, "
            "node_type = local, device_id = 0.\n"
            "Add model info, model_name = persistent_cancel_p2_pp, "
            "device_type = 0, model_path = local_context_1/controller.om.\n"
            "[udf_proxy_client.cc:158] LoadProcess:Fork udf process.\n",
            encoding="utf-8",
        )
    else:
        log.write_text(
            "select resource type is [Aarch].\n"
            "Number [1] of local udfs will untar to local_context_1.\n"
            "The time cost of host udf do untar process is [1] micro seconds.\n"
            "Add model info, model_name = persistent_cancel_p2_pp, "
            "device_type = 1, model_path = local_context_1/controller.om.\n"
            "[udf_executor_client.cc:321] LoadProcess:Fork udf process.\n",
            encoding="utf-8",
        )
    return analyze_placement([root])


def _analyze_fixture(tmp_path: Path, *, profile_delta: int = 0, target=True) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    logs = []
    calls = 0
    for index in range(3):
        path = tmp_path / f"start-{index}.log"
        calls = _write_start(path, 1001 + index)
        logs.append(path)
    before = tmp_path / "before.txt"
    before.write_text("HBM Usage Rate(%) : 5\n", encoding="utf-8")
    after = []
    for index in range(3):
        path = tmp_path / f"after-{index}.txt"
        path.write_text("HBM Usage Rate(%) : 5\n", encoding="utf-8")
        after.append(path)
    status = tmp_path / "profile-status.tsv"
    _write_profile_status(status)
    profile_root = tmp_path / "profile"
    _write_profile_tasks(profile_root, calls + profile_delta, target=target)
    profile_log = tmp_path / "profile.log"
    _write_start(profile_log, 2001, pid=99)
    profile_outputs = sum(
        line.startswith("P2_OUTPUT ")
        for line in profile_log.read_text(encoding="utf-8").splitlines()
    )
    controller = tmp_path / "driver-logs"
    _write_controller_log(controller, 99, profile_outputs)
    after_profile = tmp_path / "after-profile.txt"
    after_profile.write_text("HBM Usage Rate(%) : 5\n", encoding="utf-8")
    return analyze(
        logs,
        before,
        after,
        status,
        profile_root,
        _placement_result(tmp_path),
        profile_log=profile_log,
        controller_log_roots=[controller],
        npu_after_profile=after_profile,
    )


def test_p2_start_verifier_reconstructs_all_generations(tmp_path):
    path = tmp_path / "start.log"
    calls = _write_start(path, 1001)

    result = verify_start(path)

    assert result["pass"] is True, result["errors"]
    assert result["summary"]["aicore_calls"] == calls
    assert result["latency"]["samples"] == 1000
    assert result["type_counts"][4] == 1008


def test_p2_start_verifier_rejects_post_retirement_commit(tmp_path):
    path = tmp_path / "start.log"
    _write_start(path, 1001)
    lines = path.read_text(encoding="utf-8").splitlines()
    retirement = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("P2_OUTPUT ")
        and "type=4 " in line
        and "request=10000 " in line
    )
    committed = next(
        line
        for line in lines[:retirement]
        if line.startswith("P2_OUTPUT ")
        and "type=2 " in line
        and "request=10000 " in line
    )
    lines.insert(retirement + 1, committed)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = verify_start(path)

    assert result["pass"] is False
    assert any("commit is not owned" in error for error in result["errors"])


def test_p2_analyzer_accepts_dynamic_target_profile(tmp_path):
    result = _analyze_fixture(tmp_path)

    assert result["pass"] is True
    assert result["mechanism_pass"] is True
    assert result["profile_pass"] is True
    assert (
        result["profile"]["target_add_task_count"]
        == result["profile"]["run"]["summary"]["aicore_calls"]
    )
    assert result["profile"]["controller"]["device_executor_pids"] == [77]


def test_p2_analyzer_rejects_wrong_count_and_unrelated_add(tmp_path):
    wrong_count = _analyze_fixture(tmp_path / "count", profile_delta=-1)
    unrelated = _analyze_fixture(tmp_path / "name", target=False)

    assert wrong_count["mechanism_pass"] is True
    assert wrong_count["profile_pass"] is False
    assert unrelated["mechanism_pass"] is True
    assert unrelated["profile_pass"] is False
    assert unrelated["profile"]["target_add_task_count"] == 0


def test_p2_placement_rejects_host_local_controller(tmp_path):
    accepted = _placement_result(tmp_path)
    rejected = _placement_result(tmp_path, device=False)

    assert accepted["pass"] is True
    assert rejected["pass"] is False
    assert rejected["forbidden_evidence_counts"]["host_udf_executor"] == 1
