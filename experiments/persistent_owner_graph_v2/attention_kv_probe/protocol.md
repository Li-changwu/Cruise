# V2 KV Attention Safety Probe

This is the bounded gate before any full Decoder or P5 qualification work. It
combines the already-proven Device State Handle, B4/K384 PA-NZ update ordering,
and public FIA GraphPp path in one graph.

FunctionPp owns separate long-lived key and value buffers with shape
`[12, 4, 8, 128, 16]`. Host sends only a 20-byte sequence/slot metadata tensor.
`DevicePagedKvUpdate` writes four declared slots in place and emits a 9-int
ticket. `DeviceQueryAfterKvUpdate` consumes that ticket and forwards only the
28 KiB query, so FIA has an explicit graph dependency on the completed update.
FIA reads the same key/value Data nodes directly. A mask exposes only the four
updated logical slots, making the attention output bitwise equal to the value
written for that sequence.

The gate passes only when:

1. the public custom-op and FIA JIT toolchains build in isolated scratch;
2. AIR contains one update, one query-order primitive, and one FIA, with no
   `RefData`, `TensorMove`, or full-KV graph output;
3. ordinary Graph uses Device-placed KV inputs, returns exact compact output,
   and performs no full-KV Host input/output;
4. FunctionPp allocates the two KV buffers once, preserves both FlowMsg and
   Device address identity across two GraphPp calls, and full-scans the state;
5. each call observes exact slot-only state and an attention output equal to
   that call's newly written BF16 value;
6. Host KV input/output is 0/0 bytes and no raw address integer ABI, external
   `RefData`, private `ModelPp`, or Host `aclmdlExecute` is used.

Any failure stops this combined path. A pass remains a component result: it
does not prove zero internal Device-to-Device copies, full-model correctness,
service behavior, performance, or P5 qualification.

## r1 result

The single run `persistent-owner-v2-kv-attention-20260815-r1` failed and stops
this path. Toolchain build, AIR export, and structure passed. The exported AIR
ordered its six `Data` inputs as key, value, metadata, query, mask, and block
table, but the compile config and both callers used key, value, query,
metadata, mask, and block table. GE consequently bound BF16 query data to the
INT32 update metadata input and rejected engine assignment.

Ordinary Graph and GraphPp both stopped before a target kernel launch.
GraphPp returned compile status `1343225857` before its first Feed, so
FunctionPp performed zero allocations and zero graph calls. Address lifetime,
exact update, attention visibility, and dual-route execution were not proven.
That run did not authorize a corrected rerun or downstream P5 work.

## ABI-r2 authorization

On 2026-08-16 the user explicitly authorized one ABI-corrected experiment as a
new revision. The r1 failure and evidence remain unchanged. ABI-r2 must derive
all six input indices, AIR dtypes, and shapes from the exported graph, reject a
metadata/query swap before execution, and generate both the Graph config and
C++ input constants from that checked manifest.

Execution is staged and fail-closed: export and static ABI first; then one
ordinary Graph exact update-to-FIA run with all three target kernel launches;
then two FunctionPp calls through GraphPp with stable FlowMsg and Device
addresses, full-cache scans, exact attention, and bounded transfer-log review.
Failure at any stage stops the later stages. A complete pass remains component
evidence only and does not reopen P5 or authorize full Decoder work by itself.

## ABI-r2 result

The authorized ABI-r2 experiment ran on 2026-08-16 at clean commit `e414092`.
Export and static ABI validation passed with the required order: key, value,
metadata, query, mask, and block table. The ordinary Graph then loaded and
launched `DevicePagedKvUpdate`, `DeviceQueryAfterKvUpdate`, and
`FusedInferAttentionScore` once each, but `RunGraph` returned runtime status
`107000` (`ACL_ERROR_RT_PARAM_INVALID`) without compact outputs.

This is distinct from r1: the corrected inputs reached execution and all three
target kernels launched. Retained logs are insufficient to attribute the
runtime error to a specific operator or output boundary because failure cleanup
preserved only the first 24 sorted driver logs, not the detailed execution
tails. The bounded transfer audit found no 1.5 MiB or 3 MiB full-KV-size match.

Ordinary Graph exactness did not pass, so the protocol stopped before GraphPp.
Two-call lifetime and exactness checks, full-cache scans, and dual-route
execution remain unproven. See
`evidence/PERSISTENT-OWNER-V2-KV-ATTENTION-ABI-R2-20260816.md`. P5 remains
Stopped / Unqualified and P6 remains closed.
