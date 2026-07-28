#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    performance_source = repo / "work/g4c-attempt70b-r3-r1-b4-performance-abba"
    sys.path.insert(0, str(performance_source))
    from test_analyze_abba_performance import run_case, write_blocks

    with tempfile.TemporaryDirectory(prefix="g4-completion-selftest-") as tmp:
        root = Path(tmp)
        performance = run_case(root, "performance", write_blocks(root))
        if performance.pop("exit_status") != 0 or not performance["pass"]:
            return 10
        performance_result = root / "performance.json"
        performance_result.write_text(json.dumps(performance), encoding="ascii")
        performance_status = root / "status.tsv"
        performance_status.write_text(
            "case\texit_status\nsynthetic-performance\t0\n", encoding="ascii"
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("verify_g4_completion.py")),
            "--g4a-result",
            str(repo / "results/g4a-attempt65-compact-evidence/native/attempt65-result.json"),
            "--g4a-status",
            str(repo / "results/g4a-attempt65-compact-evidence/native/status.tsv"),
            "--g4b-result",
            str(repo / "results/g4b-attempt66b-r4/attempt66b-r4-result.json"),
            "--g4b-status",
            str(repo / "results/g4b-attempt66b-r4/status.tsv"),
            "--b2-result",
            str(repo / "results/g4c-attempt68a-b2-resident-epoch/attempt68a-result.json"),
            "--b2-status",
            str(repo / "results/g4c-attempt68a-b2-resident-epoch/status.tsv"),
            "--b4-result",
            str(repo / "results/g4c-attempt69e-r5-b4-resident-epoch-evidence/attempt69e-r5-result.json"),
            "--b4-status",
            str(repo / "results/g4c-attempt69e-r5-b4-resident-epoch-evidence/status.tsv"),
            "--recovery-result",
            str(repo / "results/g4c-attempt70a-r1-b4-recovery-evidence/attempt70a-r1-result.json"),
            "--recovery-status",
            str(repo / "results/g4c-attempt70a-r1-b4-recovery-evidence/status.tsv"),
            "--performance-result",
            str(performance_result),
            "--performance-status",
            str(performance_status),
            "--output",
            str(root / "completion.json"),
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.DEVNULL)
        if completed.returncode != 0:
            return 11

    print("g4-completion-verifier-selftest\tPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
