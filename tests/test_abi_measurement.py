import json
from pathlib import Path

import pytest

from analyze_abi_comparison import analyze
from summarize_msprof_transfers import summarize
from summarize_runtime_memcpy import summarize as summarize_runtime
from verify_abi_comparison_result import validate_result
from verify_minimal_abi_source import validate_source
from vllm_ascend_resident_epoch.abi import (
    ABI_BYTES,
    NEW_INPUTS,
    NEW_OUTPUTS,
    NEW_TOTAL_BYTES,
    OLD_INPUTS,
    OLD_OUTPUTS,
    OLD_TOTAL_BYTES,
    TOTAL_REDUCTION_BYTES,
)


SOURCE = Path(__file__).parents[1]


def test_minimal_abi_source_and_byte_ledger():
    result = validate_source(SOURCE, None)
    assert result["pass"], result
    assert ABI_BYTES["old"] == {"input": 58_720_516, "output": 78_184_928}
    assert ABI_BYTES["new"] == {"input": 260, "output": 368}
    assert OLD_TOTAL_BYTES == 136_905_444
    assert NEW_TOTAL_BYTES == 628
    assert TOTAL_REDUCTION_BYTES == 136_904_816


def test_attempt74_uses_relocatable_custom_opp_roots():
    driver = (SOURCE / "run_attempt74.sh").read_text(encoding="utf-8")
    assert 'dirname "$(dirname "${custom_set_env}")"' in driver
    assert 'dirname "$(dirname "${barrier_set_env}")"' in driver
    assert 'dirname "$(dirname "${materialize_set_env}")"' in driver
    assert 'source "${custom_set_env}"' not in driver
    assert "${materialize_opp_vendor}:${barrier_opp_vendor}:${custom_opp_vendor}" in driver


def test_msprof_summary_never_substitutes_logical_bytes(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "runtime.csv").write_text("Name,Duration\nMemcpy,10\n", encoding="utf-8")
    (new / "runtime.csv").write_text("Name,Duration\nMemcpy,8\n", encoding="utf-8")
    result = summarize(old, new)
    assert result["status"] == "not_observed"
    assert result["logical_abi_bytes_used_as_transfer"] is False
    assert result["reason"]


def test_msprof_summary_counts_only_directional_byte_rows(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "runtime.csv").write_text(
        "Name,Direction,Size(Bytes)\nMemcpy,H2D,100\nMemcpy,D2H,40\n",
        encoding="utf-8",
    )
    (new / "runtime.csv").write_text(
        "Name,Direction,Size(Bytes)\nMemcpy,H2D,20\nMemcpy,D2H,8\n",
        encoding="utf-8",
    )
    result = summarize(old, new)
    assert result["status"] == "observed"
    assert result["routes"]["old"]["total_bytes"] == 140
    assert result["routes"]["new"]["total_bytes"] == 28


def test_msprof_summary_requires_both_directions(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for root in (old, new):
        (root / "runtime.csv").write_text(
            "Name,Direction,Size(Bytes)\nMemcpy,H2D,100\n",
            encoding="utf-8",
        )
    result = summarize(old, new)
    assert result["status"] == "not_observed"
    assert result["routes"]["old"]["device_to_host_bytes"] is None


def test_msprof_summary_preserves_explicit_unavailable_reason(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()

    result = summarize(
        old,
        new,
        "CANN application profiler cannot initialize GE in the sidecar",
    )

    assert result["status"] == "not_observed"
    assert result["explicit_unavailable_reason"].startswith("CANN")
    assert "no CSV reports" in result["reason"]


def _block(route: str) -> dict:
    sizes = ABI_BYTES[route]
    samples = []
    for repetition in range(15):
        epoch_start = (repetition + 1) * 1_000_000
        samples.append(
            {
                "pass": True,
                "engine_core_step_calls": 1,
                "post_step_calls": 1,
                "socket_send_calls": 1,
                "socket_receive_calls": 1,
                "feed_calls": 1,
                "fetch_calls": 1,
                "model_calls": 2,
                "declared_input_bytes": sizes["input"],
                "declared_output_bytes": sizes["output"],
                "declared_total_bytes": sizes["input"] + sizes["output"],
                "host_control_wall_us": 100 + repetition,
                "python_cpu_us": 80 + repetition,
                "native_wall_us": 70 + repetition,
                "native_cpu_us": 60 + repetition,
                "epoch_wall_clock_start_ns": epoch_start,
                "epoch_wall_clock_end_ns": epoch_start + 100_000,
            }
        )
    return {
        "pass": True,
        "route": route,
        "repetitions": 15,
        "samples": samples,
        "resolved_classes": {"scheduler": "s", "executor": "e", "worker": "w"},
        "artifacts": {"baseline_sha256": "a" * 64},
    }


def _write_transfer_trace(
    path: Path, block: dict, route: str, *, include_runtime: bool = False
) -> None:
    input_specs = OLD_INPUTS if route == "old" else NEW_INPUTS
    output_specs = OLD_OUTPUTS if route == "old" else NEW_OUTPUTS
    rows = ["api\tpid\ttid\tstart_ns\tend_ns\tbytes\tdest_max\tkind\tstatus"]
    rows.append("rtMemcpy\t1\t1\t1\t2\t1024\t1024\t1\t0")
    for sample in block["samples"]:
        start = sample["epoch_wall_clock_start_ns"]
        for spec in input_specs:
            rows.append(
                "\t".join(
                    map(
                        str,
                        (
                            "FeedDataFlowGraphTensor",
                            1,
                            1,
                            start + 10,
                            start + 20,
                            spec.nbytes,
                            spec.nbytes,
                            -1,
                            0,
                        ),
                    )
                )
            )
        if include_runtime:
            rows.append(
                f"rtMemcpyEx\t1\t1\t{start + 30}\t{start + 40}\t100\t100\t1\t0"
            )
            rows.append(
                f"rtsMemcpyAsync\t1\t1\t{start + 50}\t{start + 60}\t40\t40\t2\t0"
            )
        for spec in output_specs:
            rows.append(
                "\t".join(
                    map(
                        str,
                        (
                            "FetchDataFlowGraphTensor",
                            1,
                            1,
                            start + 70,
                            start + 80,
                            spec.nbytes,
                            spec.nbytes,
                            -1,
                            0,
                        ),
                    )
                )
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_analyzer_and_independent_verifier_recompute_dataflow_payload(tmp_path):
    routes = ("old", "new", "new", "old")
    paths = []
    traces: dict[str, Path] = {}
    results: dict[str, Path] = {}
    for index, route in enumerate(routes):
        label = ("old-1", "new-1", "new-2", "old-2")[index]
        block = _block(route)
        path = tmp_path / f"block-{index}.json"
        path.write_text(json.dumps(block), encoding="utf-8")
        paths.append(path)
        trace = tmp_path / f"{label}.tsv"
        _write_transfer_trace(trace, block, route)
        traces[label] = trace
        results[label] = path
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({"pass": True}), encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"pass": True}), encoding="utf-8")
    profiler = tmp_path / "profiler.json"
    profiler.write_text(
        json.dumps(
            {
                "status": "not_observed",
                "reason": "fixture has no physical profiler fields",
                "logical_abi_bytes_used_as_transfer": False,
                "routes": {},
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(summarize_runtime(traces, results, tmp_path / "filtered")),
        encoding="utf-8",
    )
    result = analyze(paths, semantic, source, profiler, runtime)
    assert result["pass"], result
    assert result["transfer_trace"]["routes"]["old"]["dataflow_total_bytes_median"] == 136_905_444
    assert result["transfer_trace"]["routes"]["new"]["dataflow_total_bytes_median"] == 628
    assert result["transfer_trace"]["routes"]["old"]["runtime_memcpy_statuses"] == [
        "observed_zero"
    ]
    assert result["observed_dataflow_payload"]["reduction_bytes_per_epoch"] == 136_904_816
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    summary = validate_result(result, require_artifacts=True)
    assert summary["pass"] is True
    assert summary["samples_per_route"] == 30
    assert summary["runtime_memcpy_statuses"] == ["observed_zero"]

    filtered = tmp_path / "filtered" / "old-1.tsv"
    filtered.write_text(
        filtered.read_text(encoding="utf-8")
        + "old-1\t0\truntime_memcpy\thost_to_device\trtMemcpy\t1\t1\t2\t3\t1\t1\t1\t0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="filtered transfer trace SHA256 mismatch"):
        validate_result(result, require_artifacts=True)


def test_transfer_trace_rejects_epoch_boundary_crossing(tmp_path):
    traces: dict[str, Path] = {}
    results: dict[str, Path] = {}
    labels = ("old-1", "new-1", "new-2", "old-2")
    for label, route in zip(labels, ("old", "new", "new", "old"), strict=True):
        block = _block(route)
        result_path = tmp_path / f"{label}.json"
        result_path.write_text(json.dumps(block), encoding="utf-8")
        trace_path = tmp_path / f"{label}.tsv"
        _write_transfer_trace(trace_path, block, route)
        traces[label] = trace_path
        results[label] = result_path
    old_trace = traces["old-1"]
    with old_trace.open("a", encoding="utf-8") as stream:
        stream.write("rtMemcpy\t1\t1\t999990\t1000010\t1\t1\t1\t0\n")

    result = summarize_runtime(traces, results, tmp_path / "filtered")

    assert result["status"] == "not_observed"
    assert result["blocks"]["old-1"]["boundary_crossing_calls"] == 1


def test_transfer_trace_rejects_unknown_interposed_api(tmp_path):
    traces: dict[str, Path] = {}
    results: dict[str, Path] = {}
    labels = ("old-1", "new-1", "new-2", "old-2")
    for label, route in zip(labels, ("old", "new", "new", "old"), strict=True):
        block = _block(route)
        result_path = tmp_path / f"{label}.json"
        result_path.write_text(json.dumps(block), encoding="utf-8")
        trace_path = tmp_path / f"{label}.tsv"
        _write_transfer_trace(trace_path, block, route)
        traces[label] = trace_path
        results[label] = result_path
    new_trace = traces["new-1"]
    contents = new_trace.read_text(encoding="utf-8")
    new_trace.write_text(
        contents.replace("FeedDataFlowGraphTensor", "DataFlowFuture", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid transfer trace API"):
        summarize_runtime(traces, results, tmp_path / "filtered")


def test_transfer_trace_reports_runtime_memcpy_when_present(tmp_path):
    traces: dict[str, Path] = {}
    results: dict[str, Path] = {}
    labels = ("old-1", "new-1", "new-2", "old-2")
    for label, route in zip(labels, ("old", "new", "new", "old"), strict=True):
        block = _block(route)
        result_path = tmp_path / f"{label}.json"
        result_path.write_text(json.dumps(block), encoding="utf-8")
        trace_path = tmp_path / f"{label}.tsv"
        _write_transfer_trace(trace_path, block, route, include_runtime=True)
        traces[label] = trace_path
        results[label] = result_path

    result = summarize_runtime(traces, results, tmp_path / "filtered")

    assert result["status"] == "observed"
    assert result["blocks"]["new-1"]["runtime_memcpy_status"] == "observed"
    assert result["routes"]["new"]["runtime_memcpy_records_median"] == 2
    assert result["routes"]["new"]["runtime_memcpy_total_directional_bytes_median"] == 140


def test_transfer_trace_requires_complete_dataflow_payload(tmp_path):
    traces: dict[str, Path] = {}
    results: dict[str, Path] = {}
    labels = ("old-1", "new-1", "new-2", "old-2")
    for label, route in zip(labels, ("old", "new", "new", "old"), strict=True):
        block = _block(route)
        result_path = tmp_path / f"{label}.json"
        result_path.write_text(json.dumps(block), encoding="utf-8")
        trace_path = tmp_path / f"{label}.tsv"
        _write_transfer_trace(trace_path, block, route)
        traces[label] = trace_path
        results[label] = result_path
    trace = traces["new-1"]
    rows = trace.read_text(encoding="utf-8").splitlines()
    del rows[2]
    trace.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = summarize_runtime(traces, results, tmp_path / "filtered")

    assert result["status"] == "not_observed"
    assert result["blocks"]["new-1"]["dataflow_tensor_payload_status"] == "not_observed"
