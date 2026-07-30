# M0 Repeated Lifecycle Acceptance on NPU0-1

Date: 2026-07-31 (Asia/Shanghai)

Status: accepted. This evidence closes M0 and qualifies profile
`attempt74-910b2-cann900-npu01-r1` as
`m0-qualified-developer-preview`. It does not close M1 or claim Stable v1.0.

## Qualification Boundary

Both cycles ran through SSH target `NPU0-1`, Conda environment
`vllm-hust-dev`, and physical NPU 0 from clean Cruise commit
`8408a689ea0827ef39b060635e626e61a8a95d5a`. No model was downloaded, copied
from another server, or moved. The pre-existing frozen model inputs remained
under `/dev/shm/cruise-m0-assets-npu01-r1`. Both cycles reused the single
342-file, 15,231,237,408-byte content-addressed runtime-weight bundle under
`/workspace/cruise-assets`; wheel, native build, relocated AIR, controller,
configuration, cache, logs, socket, and GraphPp output were rebuilt below an
independent per-cycle `/dev/shm` prep or run directory.

Each cycle executed:

```text
clean commit -> build wheel/native/AIR -> install -> smoke
  -> NPU 0 doctor -> runtime doctor --deep -> real EngineCore
  -> post-cleanup verifier -> cleanup -> uninstall -> residue audit
```

## Results

| Check | Cycle 1 | Cycle 2 |
|---|---:|---:|
| wheel bytes | 51,435 | 51,435 |
| wheel SHA256 | `0f083327...ee67` | `92a07ef2...f99f` |
| relocated AIR SHA256 | `5ab48ca6...62ab` | `2a234520...a76` |
| sidecar SHA256 | `55190f4f...f8b` | `55190f4f...f8b` |
| AIR FileConstants rewritten/audited | 342/342 | 342/342 |
| EngineCore cases | 13/13 | 13/13 |
| Feed/Fetch calls | 13/13 | 13/13 |
| device model calls | 49 | 49 |
| retained structured evidence | 88 KiB | 88 KiB |

Both post-cleanup verifiers returned `pass=true`, confirmed the expected
resident scheduler, worker, and UniProc executor, and confirmed distributed
state was destroyed. Every case matched the frozen greedy token baseline and
scheduler accounting for batch sizes 1, 2, and 4 and K values 1, 2, 4, and 8,
including the independent-EOS case.

After each cycle, the wheel was absent from `vllm-hust-dev`; the managed
scratch root, resident socket, sidecar, NPU executor, and UDF executor were
absent; `/workspace/Cruise` was clean; and persistent output was below the
100 MiB limit. This firmware reports that `npu-smi info proc -i 0` is not
supported, so the post-stop process audit used exact system process names.
The installed doctor had already established that physical NPU 0 was healthy
and idle before each start.

## Defects Found During Acceptance

The first candidate run stopped before graph execution because Cruise exported
an `ASCEND_CACHE_PATH` whose final directory did not yet exist. CANN 9.0.0
returned `EC0006`. Commit `56dc909` now creates every exported managed cache,
log, temporary, and GraphPp directory before starting the child, with a
regression test.

The next candidate executed all 13 cases but the old verifier incorrectly
required a `native_library` in pure sidecar mode and required the already
cleaned GraphPp directory to remain. Commit `8408a68` records and validates the
persistent runtime-weight bundle, treats the native library as optional when a
native server is present, and permits per-run GraphPp output to disappear after
successful cleanup. The complete remote suite then passed 110 tests.

## Evidence Files

The exact verifier output, runner result, doctor reports, runtime configuration,
AIR relocation report, source commit, and artifact hashes are committed under
[`m0-lifecycle-npu01-cycle1/`](m0-lifecycle-npu01-cycle1/) and
[`m0-lifecycle-npu01-cycle2/`](m0-lifecycle-npu01-cycle2/). Raw runtime logs,
compiled artifacts, AIR files, and weights are intentionally excluded.

## Gate Decision

All five M0 items and the repeated lifecycle exit condition are satisfied. The
next ordered work is M1: rerun the 1,000-request EngineCore differential on
NPU0-1 physical NPU 0, then run the OpenAI API semantic gate. No M1 completion
claim is made by this evidence.
