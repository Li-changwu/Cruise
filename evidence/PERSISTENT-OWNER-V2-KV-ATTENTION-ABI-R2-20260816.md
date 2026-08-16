# Controller-Aware Graph V2 KV attention ABI-r2 stop record

The authorized ABI-r2 experiment is bound to clean source commit
`e414092e1d58a07dac21dfa18fde7aefb6896c58`. It used two staged runs:

- `persistent-owner-v2-kv-attention-abi-r2-export-20260816-r1`;
- `persistent-owner-v2-kv-attention-abi-r2-graph-20260816-r1`.

Export and static ABI validation passed. Ordinary Graph execution failed, so
the fail-closed protocol did not run GraphPp, full Decoder integration, a P5
Graph/Owner pair, or the six-start matrix. This is a failed component gate, not
a P5 performance result. The separate ABI-r1 negative result remains unchanged
in `PERSISTENT-OWNER-V2-KV-ATTENTION-20260815.md`.

## ABI correction

ABI-r2 derives the six public inputs from AIR graph connectivity and validates
their exact index, dtype, and shape before execution:

| Index | Role | AIR dtype | Shape |
| --- | --- | --- | --- |
| 0 | key cache | `DT_BF16` | `[12, 4, 8, 128, 16]` |
| 1 | value cache | `DT_BF16` | `[12, 4, 8, 128, 16]` |
| 2 | metadata | `DT_INT32` | `[5]` |
| 3 | query | `DT_BF16` | `[4, 28, 1, 128]` |
| 4 | mask | `DT_BOOL` | `[4, 1, 1, 384]` |
| 5 | block table | `DT_INT32` | `[4, 3]` |

The checked manifest generates both Graph tensor descriptors and C++ input
constants. It maps AIR `DT_BF16` to the Graph API spelling `DT_BFLOAT16` and
rejects a metadata/query swap before hardware execution.

Static validation passed: all 32 V2 tests, the 200-test CI-equivalent set, and
271 repository tests excluding the separately known-broken
`tests/test_real_scheduler.py`. Python and Bash syntax, Host C++, AArch64
FunctionPp C++, the minimal ABI verifier, and repository payload audit also
passed.

## Export stage

The export-only run passed the six-input ABI and graph-structure checks. It
contained one `DevicePagedKvUpdate`, one `DeviceQueryAfterKvUpdate`, one
`FusedInferAttentionScore`, and compact outputs. The AIR SHA-256 was
`1247f82256e1136b3d4a255d19a8d6231dbf5b4aca8cfad028dd6c778b7285a4`;
the graph-text SHA-256 was
`81e9e934d8f240ba11840dcc94e20cdb7a1f461bedcc40be5a83bfdeb4b2257d`.

The exporter retained its historical post-result teardown status 139, but the
complete `pass=true` result and all required artifacts existed, and the staged
driver exited 0. No Graph or GraphPp execution occurred in this run.

## Ordinary Graph stage

The Graph run loaded a valid AIR, added the graph successfully, and prepared
all six Device-placed inputs from the checked ABI. That run independently
exported an AIR with SHA-256
`8ef964b9b04ce6e639f3c30b2bf78ee380d0dd76d8b796e20e9085d5a5065214`
and graph text with SHA-256
`45a07bbc125eae40d7df2c0c5a538fb28872e1a90e2c0b84202899bc1f0d7e71`.
Runtime launch records show exactly one launch of each required kernel:

1. `DevicePagedKvUpdate`;
2. `DeviceQueryAfterKvUpdate`;
3. `FusedInferAttentionScore`.

`session->RunGraph` then returned `107000`, which the installed public ACL
header defines as `ACL_ERROR_RT_PARAM_INVALID`. No compact output was returned,
so ticket and attention exactness did not pass. This is later than ABI-r1,
which failed before any target launch, and proves that the metadata/query order
was corrected. It does not prove which operator or output-recovery boundary
produced the runtime error.

The bounded transfer audit retained 1,526 records and found no transfer whose
size matched one 1.5 MiB cache or the combined 3 MiB key/value state. The Graph
record therefore reports zero matching full-KV Host transfers. This does not
prove general zero-copy execution or compensate for the missing exact output.

## Confidence boundary

The failure cleanup copied only the first 24 driver-log files in sorted order.
Those files primarily cover export and compilation; the detailed host and
Device log tails for the actual Graph execution process were removed with
scratch. The retained evidence supports only this statement: runtime returned
an invalid-parameter status after all three target kernels had been launched.
It does not support assigning the cause specifically to KV update, dependency
forwarding, FIA execution, synchronization, or compact-output recovery.

The Graph driver exited 10. NPU 0 recovered from 3,452 MiB to 3,453 MiB HBM,
reported no visible process, and had zero `/dev/shm` policy violations. No NPU
reset was performed.

## Decision boundary

The ordered protocol required an exact ordinary Graph result before two
FunctionPp calls through GraphPp. Ordinary Graph did not meet that condition,
so GraphPp was not run. Address stability, two-call exactness, full-cache scans,
and Graph/GraphPp dual-route execution therefore remain unproven.

P5 remains Stopped / Unqualified and P6 remains closed. A further hardware run
requires a separately authorized diagnostic revision that first fixes complete
execution-log retention and narrows the `107000` source; this result does not
authorize full Decoder or performance work.
