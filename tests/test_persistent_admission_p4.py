import json
from pathlib import Path

from experiments.persistent_admission_p4.aggregate_p4_gate import aggregate
from experiments.persistent_admission_p4.verify_p4_gate import verify


ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "experiments" / "persistent_admission_p4"


def _state(index: int, *, tag: str = "primary", commits: int = 256) -> dict:
    return {
        "request": 20000 + index,
        "generation": index + 1,
        "tag": tag,
        "row": index % 4,
        "send_count": 1,
        "rejected_admissions": 0,
        "commits": commits,
        "stream_messages": commits,
        "retired": True,
        "cancelled": False,
        "final_position": 128 + commits - 1,
        "final_page": (128 + commits - 1) // 128,
        "final_checksum": 123,
        "finish_reason": 2,
        "tokens": [11] * commits,
        "token_timestamps_ns": list(range(commits)),
    }


def test_p4_workload_freezes_the_formal_service_case():
    workload = json.loads((P4 / "workload.json").read_text(encoding="utf-8"))
    primary = workload["primary"]

    assert primary == {
        "name": "primary-c4",
        "prompt_tokens": 128,
        "output_tokens": 256,
        "request_count": 32,
        "closed_loop_concurrency": 4,
        "ignore_eos": True,
        "streaming": True,
    }
    assert workload["regression"]["overload_concurrency"] == 8
    assert workload["regression"]["burst_gap_ms"] == 100


def test_p4_reuses_p3_owner_and_keeps_host_control_request_bounded():
    host = (P4 / "persistent_admission_p4_host.cpp").read_text(encoding="utf-8")
    runner = (P4 / "run_on_910b.sh").read_text(encoding="utf-8")
    protocol = (P4 / "protocol.md").read_text(encoding="utf-8")

    assert 'FunctionPp("persistent_decoder_p3_pp")' in host
    assert 'FlowNode("persistent_decoder_p3_node", 1, 1)' in host
    assert 'controller_source=${p3_dir}/controller' in runner
    assert '"${controller_source}/persistent_decoder_p3.cpp"' in runner
    assert "feed_calls == expected_feeds" in host
    assert 'scenario == "primary-c4" ? 33 : 36' in host
    assert "event.reserved[0] = cohort_id" in host
    controller = (
        ROOT
        / "experiments"
        / "persistent_decoder_p3"
        / "controller"
        / "persistent_decoder_p3.cpp"
    ).read_text(encoding="utf-8")
    assert "candidate.staged = false" in controller
    assert "if (!row.active || row.staged) return false;" in controller
    assert "Host request admission and asynchronous output drain" in protocol
    assert "no token-step" in protocol


def test_p4_runner_confines_intermediates_and_audits_tmpfs():
    runner = (P4 / "run_on_910b.sh").read_text(encoding="utf-8")

    assert "cruise-p4-service-${physical_npu}-$$" in runner
    assert 'python3 "${lifecycle_tool}" shm-audit --summary-only' in runner
    assert 'STORAGE_GUARD_PROJECT_AUDIT_INTERVAL_SECONDS=2' in runner
    assert 'cd -- "${scratch}"' in runner
    assert 'TMPDIR="${tmp}"' in runner
    assert "storage_guard_cleanup_scratch" in runner
    assert '! -f "${evidence}/summary.json"' in runner
    assert '"${evidence}/source-identity.sha256"' in runner


def test_primary_gate_requires_exact_oracle_recurrence():
    reference = _state(0)
    oracle = {
        "pass": True,
        "scenario": "b1-full",
        "model_calls": 383,
        "request_states": [reference],
    }
    summary = {
        "gate": "P4-SERVICE",
        "pass": True,
        "scenario": "primary-c4",
        "feed_calls": 33,
        "aicore_calls": 3064,
        "total_commits": 8192,
        "total_retired": 32,
        "single_token_outputs": 8192,
        "max_pending_requests": 4,
        "max_active_rows": 4,
        "rejected": 0,
        "protocol_errors": 0,
        "shutdown": True,
        "request_states": [_state(index) for index in range(32)],
    }

    result = verify(summary, oracle)

    assert result["pass"] is True
    summary["request_states"][17]["tokens"][-1] = 12
    result = verify(summary, oracle)
    assert result["pass"] is False
    assert any(item.get("request_index") == 17 for item in result["mismatches"])


def test_p4_protocol_keeps_phase_claims_separate():
    protocol = (P4 / "protocol.md").read_text(encoding="utf-8")

    assert "not a P5 Host CPU, TPOT, throughput, or TTFT qualification" in protocol
    assert "does not claim that Prefill and Decode run concurrently" in protocol
    assert "P6" in protocol
    assert "110 through 131 AICore calls" in protocol
    assert "P4 is Candidate Hardware Validated" in protocol


def test_p4_aggregate_requires_matching_source_identity(tmp_path):
    primary = tmp_path / "primary"
    regression = tmp_path / "regression"
    for root, scenario in ((primary, "primary-c4"), (regression, "regression")):
        evidence = root / "evidence"
        evidence.mkdir(parents=True)
        (evidence / "summary.json").write_text(
            json.dumps(
                {
                    "pass": True,
                    "scenario": scenario,
                    "requests": 1,
                    "feed_calls": 2,
                    "aicore_calls": 3,
                    "total_commits": 4,
                }
            ),
            encoding="utf-8",
        )
        (evidence / "gate.json").write_text(
            json.dumps({"pass": True, "device_decode_coverage_percent": 100}),
            encoding="utf-8",
        )
        (evidence / "source-identity.sha256").write_text(
            "same identity\n", encoding="utf-8"
        )
    (primary / "evidence" / "oracle-identity.sha256").write_text(
        "oracle identity\n", encoding="utf-8"
    )

    result = aggregate(primary, regression)

    assert result["pass"] is True
    (regression / "evidence" / "source-identity.sha256").write_text(
        "different identity\n", encoding="utf-8"
    )
    assert aggregate(primary, regression)["pass"] is False
