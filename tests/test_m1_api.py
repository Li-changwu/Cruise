import json
from pathlib import Path
import sys

from experiments.m1_exit.run_api_differential import (
    ApiManifest,
    _bounded_logger_command,
    _server_command,
    _with_tokenizer_override,
    compare_results,
    load_manifest,
)


MANIFEST = (
    Path(__file__).parents[1] / "experiments" / "m1_exit" / "api_cases.json"
)


def test_api_manifest_covers_required_semantics():
    manifest = load_manifest(MANIFEST)
    cases = {case.name: case for case in manifest.cases}

    assert len(cases) == 8
    assert cases["nonstream-batch"].choice_count == 2
    assert cases["stream-batch"].choice_count == 2
    assert cases["eos-second-token"].stop_token_ids == (2776,)
    assert cases["eos-second-token"].expected_token_count == 2
    assert cases["eos-second-token"].expected_finish_reason == "stop"
    assert cases["unsupported-min-tokens"].min_tokens == 1
    assert cases["disconnect-after-device"].disconnect_after_tokens == 3
    assert all(
        len(tokens) + case.max_tokens - 1 <= 8
        for case in manifest.cases
        for tokens in (
            case.prompt if isinstance(case.prompt[0], list) else [case.prompt]
        )
    )


def test_api_comparison_ignores_transport_identity_but_not_tokens(tmp_path):
    comparable = {
        "pass": True,
        "cases": [
            {
                "name": "stream-single",
                "stream": True,
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [1, 2],
                        "finish_reason": "length",
                        "stop_reason": None,
                    }
                ],
                "usage": {"completion_tokens": 2},
                "done": True,
                "disconnected": False,
            }
        ],
    }
    baseline = tmp_path / "baseline.json"
    cruise = tmp_path / "cruise.json"
    baseline.write_text(json.dumps(comparable), encoding="utf-8")
    cruise.write_text(json.dumps(comparable), encoding="utf-8")
    assert compare_results(baseline, cruise)["pass"]

    changed = json.loads(json.dumps(comparable))
    changed["cases"][0]["choices"][0]["token_ids"] = [1, 3]
    cruise.write_text(json.dumps(changed), encoding="utf-8")
    result = compare_results(baseline, cruise)
    assert not result["pass"]
    assert "stream-single" in result["mismatches"]


def test_api_server_log_is_bounded(tmp_path):
    output = tmp_path / "api-server-cruise.log"
    command, metadata = _bounded_logger_command(output)

    assert Path(command[1]).parts[-2:] == ("storage_guard", "bounded_log.py")
    assert command[command.index("--head-bytes") + 1] == str(4 * 1024 * 1024)
    assert command[command.index("--tail-bytes") + 1] == str(4 * 1024 * 1024)
    assert metadata == tmp_path / "api-server-cruise.meta.json"


def test_api_server_uses_current_environment_console_script(tmp_path):
    manifest = ApiManifest("cruise-m1", tmp_path, ())
    command = _server_command(
        mode="baseline", model=tmp_path / "model", manifest=manifest, port=8000
    )

    assert Path(command[0]) == Path(sys.executable).with_name("vllm")
    assert command[1] == "serve"
    assert "vllm.entrypoints.cli.main" not in command
    assert "--no-async-scheduling" in command


def test_api_tokenizer_can_be_overridden_without_changing_manifest(
    tmp_path, monkeypatch
):
    manifest = load_manifest(MANIFEST)
    tokenizer = tmp_path / "tokenizer"
    monkeypatch.setenv("CRUISE_API_TOKENIZER", str(tokenizer))

    overridden = _with_tokenizer_override(manifest)

    assert overridden.tokenizer == tokenizer
    assert overridden.served_model_name == manifest.served_model_name
    assert overridden.cases == manifest.cases
