# Operations and Failure Boundary

## Startup Sequence

1. `cruise smoke` validates the installed package without importing NPU
   libraries.
2. `cruise doctor --mode npu` validates the exact profile and selected device.
3. `cruise doctor --mode runtime` validates every runtime path and configured
   integrity record.
4. `cruise run` creates one marked run directory, sources CANN, exports the
   validated internal environment, and starts the child process group.
5. The Python backend starts the sidecar, waits for its socket and initialization
   response, then performs the existing one-step warmup before serving epochs.

No model execution is attempted after a failed doctor or runtime validation.

## Shutdown and Cleanup

SIGINT and SIGTERM are forwarded to the whole child process group. The worker
requests a sidecar shutdown; timeout escalation terminates and then kills only
that process group. The per-run directory is removed only when its parent and
marker match the directory created by Cruise. External weights are required to
live outside that cleanup tree.

The shared scratch root is retained as an empty marked directory so another
run can reuse it. `cruise cleanup --config ...` removes it only when no other
entry exists.

## Diagnostic Levels

| Command | Expected use |
|---|---|
| `cruise smoke --json` | package/CI check on any machine |
| `cruise doctor --mode source --json` | installed contract report |
| `cruise doctor --mode npu --device N --json` | software and NPU readiness |
| `cruise doctor --mode runtime --config FILE --json` | fast pre-start asset check |
| `cruise doctor --mode runtime --config FILE --deep --json` | deployment qualification with full weight hashing |

The JSON output is stable schema 1 and is suitable for automation. A warning
does not change the exit code; any failed check returns nonzero.

## Failure Interpretation

| Failure | Required action |
|---|---|
| Compatibility or contract mismatch | Install the exact profile or create a separately validated profile |
| Missing/wrong asset hash | Rebuild or recopy the asset; never bypass the check |
| OPP layout failure | Repair the vendor root before model load |
| Insufficient `/dev/shm` | Remove only known marked scratch or choose another supported machine |
| Sidecar startup timeout/exit | Inspect bounded stdout/CANN logs from the current run; do not Host replay |
| Device error after execution begins | Treat the request as ambiguous; do not duplicate token/KV advancement |
| Unmarked scratch content | Investigate ownership; Cruise intentionally refuses deletion |

The sidecar protocol reports an explicit `prepared -> executing -> committed`
state. Host replay is permitted only when the response is `prepared` and every
request in the plan was still Host-owned before the epoch. A Feed/Fetch error,
socket interruption after submission, invalid committed output, or any batch
containing a device-owned row is fail-stop; the scheduler must not duplicate
token or KV advancement. The dedicated resident worker has no stock Host model
to replay and therefore remains fail-stop even for a prepared error.

## Logging and Data Handling

Runtime caches, CANN logs, sockets and temporary output are placed below the
per-run `/dev/shm` directory. Prompt text, generated text, credentials and raw
model tensors are not intentionally logged by the Cruise control layer. Formal
profiling remains opt-in and must follow `docs/REPOSITORY_POLICY.md`; it is not
a normal product-health mechanism.
