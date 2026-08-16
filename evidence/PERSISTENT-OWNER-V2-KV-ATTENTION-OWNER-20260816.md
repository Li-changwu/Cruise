# Controller-Aware Graph V2 owner-only component pass

The bounded owner-only run
`persistent-owner-v2-kv-attention-owner-only-20260816-r2` passed on NPU 0 on
2026-08-16. It proves that FunctionPp can allocate and retain the KV state in
its CANN execution ownership domain and pass the same FlowMsg objects directly
to GraphPp for two exact update-to-attention calls.

Ordinary Graph was intentionally not run. Its Host-created raw Device tensor
path remains the frozen ABI-r2 negative diagnostic; it is not a prerequisite
for the FunctionPp ownership contract. This is a component result, not a full
Decoder, service, performance, P5, or P6 result.

## E0: export and static ABI

The exported AIR passed the structural gate with six inputs in this order:

| Index | Role | AIR dtype | Shape |
| --- | --- | --- | --- |
| 0 | key cache | `DT_BF16` | `[12, 4, 8, 128, 16]` |
| 1 | value cache | `DT_BF16` | `[12, 4, 8, 128, 16]` |
| 2 | metadata | `DT_INT32` | `[5]` |
| 3 | query | `DT_BF16` | `[4, 28, 1, 128]` |
| 4 | mask | `DT_BOOL` | `[4, 1, 1, 384]` |
| 5 | block table | `DT_INT32` | `[4, 3]` |

It contains one `DevicePagedKvUpdate`, one
`DeviceQueryAfterKvUpdate`, one `FusedInferAttentionScore`, no `RefData`, no
`TensorMove`, one explicit update-ticket-to-FIA dependency, and no full-KV
graph output. The AIR SHA-256 is
`c438dd85bd05f9a64fd799eba8143b4549450dde4603a158cf70e7a6b3b0a906`.

The exporter wrote a complete `pass=true` result and all checked artifacts,
then retained its historical cleanup-time status 139. The driver accepts 139
only after independently confirming the result and required files. This
cleanup defect remains visible in `export-status.tsv`; it did not occur during
GraphPp compilation, Feed, Fetch, or result verification.

## E1-E2: owner execution and exactness

The DataFlow graph add, compile, and model load statuses were all zero. Both
Feed and Fetch rounds returned zero. FunctionPp called
`RunFlowModel("attention_kv_graph_0")` twice and returned exact summaries for
sequence 1 and sequence 2.

Both full key and value cache scans passed after each call. Only the four
declared slots changed, and the second call preserved the first call's state.
Attention observed BF16 1.0 on the first call and BF16 2.0 on the second call.
Both update tickets, the mask, block table, and query were exact.

## E3: ownership and transfer audit

- `allocation_count=1` and `graph_call_count=2`;
- FlowMsg identity and Device address stayed stable across both calls;
- the FunctionPp-owned key/value allocation was 3 MiB;
- Host KV input/output was 0/0 bytes;
- Host sent 40 bytes of metadata and received 512 bytes of summaries in total;
- no raw-address integer ABI or external `RefData` was used;
- update, order, and FIA each had one task-sink registration record;
- 1,524 bounded transfer-log records contained no 1.5 MiB or 3 MiB full-cache
  size match.

Task-sink registration is load-time evidence, not a per-Feed launch counter.
The two semantic executions are established by the successful Feed/Fetch
rounds, `graph_call_count=2`, and the two exact controller summaries. Absence
of a full-cache size in bounded runtime logs is not a universal zero-copy
proof.

## E4: identity and cleanup

The execution is bound directly to clean source commit
`93aa31c6265145c35da669bb5fac1bd5e6b3d8a4`; the captured source-worktree
status is empty. The compact identity record also binds the custom package,
FIA package manifest, AIR, ABI, Graph, FunctionPp, and deploy configs.

The driver exited 0. NPU 0 recovered from 3,451 MiB to 3,452 MiB HBM with three
stable samples and no visible process. No NPU reset was performed. Static
verification passed 33 focused tests and 272 repository tests excluding the
separately isolated `tests/test_real_scheduler.py`, plus Bash/Python syntax
checks and `git diff --check`.

Compact evidence is retained under
`evidence/persistent-owner-v2-kv-attention-owner-20260816/`. The complete
seven-day diagnostic bundle is under
`/workspace/cruise-runs/persistent-owner-v2-kv-attention-owner-only-20260816-r2`.

## Decision boundary

E0-E4 pass for the bounded FunctionPp-owned update-to-FIA component. This
removes the cross-process KV ownership blocker for this component only. P5
remains Stopped / Unqualified and P6 remains closed. A separate exploration
must integrate and verify the 28-layer Decoder before any new Graph/Owner
performance pair.
