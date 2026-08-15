# P3 Persistent Decoder protocol

P3 ports the pinned Qwen2.5-7B-Instruct greedy Decoder onto the P2 Persistent
Device Model Owner. It proves decoder semantics and Device KV ownership; it is
not a service-performance result and does not qualify Cruise against vLLM.

## Allocator diagnostic

Before changing the Decoder cache ABI, `allocator_probe/` isolates the CANN
FunctionPp allocation paths without loading the Decoder GraphPp. Separate
processes probe 21, 28, and 42 MiB BF16 tensors, `FlowBufferFactory` wrapping,
raw messages, and two- or four-tensor 21 MiB groups. Pair and four-way cases
retain 84 MiB at once to distinguish a per-message limit from an aggregate
FunctionPp budget. Each observation validates message metadata, shape, dtype,
element count, byte size, pointer, and first and last writable bytes. A failed
case is evidence about that allocation path and concurrency only; it is not
evidence that P3 correctness or performance qualifies.

## Frozen identity

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
- Tensor parallelism and pipeline parallelism: `TP=1`, `PP=1`.
- Sampling: greedy, one primary EOS token, with `ignore_eos` permitted only in
  the declared full-budget case.
- Static Device rows: four.
- KV page size: 128 tokens.
- Lease: three pages per admitted row, 384 tokens total.

## Ordered sub-gates

1. `P3-QK384` proves the Device QK MatMul accepts the 384-token attention
   extent and matches a BF16 matrix-multiply reference. The explicit tiling is
   retained as evidence and may not be inferred from the old eight-token AIR.
2. `P3-AIR384` exports one B=4 decoder step with twelve physical KV pages,
   validates its ABI, and proves exact eager recurrence across page boundaries.
3. `P3-OWNER` installs that AIR as the Persistent Owner's invoked compute
   closure. One admission carries the bounded prompt and complete KV lease;
   the Device advances prompt position, Decode position, page cursor, KV state,
   greedy sampling, output commits, EOS, cancellation, and retirement.
4. `P3-GATE` compares three independent Owner starts with a cold Graph oracle
   for B=1 short, B=1 128-prompt/256-output, and B=4 mixed cases.

The Host may send admission, credit, cancel, and shutdown events and may drain
Committed Prefix outputs asynchronously. It may not send a token-step command,
select a Device Scheduling Quantum, advance a Decode position or page cursor,
or retain authoritative per-token KV state. P3 serializes prompt ingestion and
Decode inside the Owner at Device Scheduling Quantum boundaries; it makes no
concurrent Prefill/Decode claim.

## P3-QK384 gate

The probe uses Q shape `[1, 28, 128]`, K shape `[28, 128, 384]`, and output
shape `[1, 28, 384]`. It must prove:

- the standard NPU MatMul executes on the target NPU;
- output is bit-exact to `torch.matmul` in BF16, or any accepted tolerance is
  explicitly recorded before decoder export;
- the 72-byte decoder tiling record declares `n=384`, covers every N tile, and
  is retained as QK capacity evidence rather than exposed as a Decoder AIR
  input; the standard NPU MatMul owns its runtime tiling;
- the target NPU returns to the pre-run HBM baseline and has no visible owner.

## P3 final gate

For every declared row, the Graph oracle and Persistent Owner must match token
IDs, finish reason, EOS position, final Device position, page cursor, live KV
checksum, and Committed Prefix. Device Decode coverage must be 100%. Admission
and shutdown may each use one Host Feed; Feed count may not scale with prompt or
output token count. P2 owner identity, cancellation, generation isolation,
retirement-before-reuse, Output Credit, and asynchronous output drain contracts
remain mandatory.

## Current validated state

P3 is Candidate Hardware Validated on physical NPU 0. B=1 short, B=1 full,
natural primary EOS, and B=4 mixed recurrence passed structured Owner-versus-
cold-Graph comparison. The final B=4 candidate passed three independent cold
Owner starts with byte-identical summaries and comparisons; the aggregate gate
is `persistent-decoder-p3-three-start-gate-20260814T114811Z`. A separate
delayed-drain run completed 256 outputs, and the real-Decoder lifecycle run
passed cancellation, Output Credit, duplicate/stale/generation isolation,
retirement, and row-reuse checks.

The first run named `persistent-decoder-p3-oracle-b1-eos-20260814T095946Z` is
non-gating: it exhausted eight output tokens without emitting EOS. The formal
oracle now requires exactly one committed token equal to the pinned primary EOS
`151645` and `finish_reason=1`; the corrected Owner and oracle passed. This
prevents a successful process exit from being mistaken for EOS semantics.

The runtime AIR SHA-256 is
`53e7c8407ac638af4c8b50865e8cc1c5ea428aedfa399b56c9bf014b792cc60b`.
The cumulative external-weight bundle has 721 logical files and identity
`f93a2828d8b3ded10acc3831d114abe8b57738ad3adfc357a0f2bb047939866b`.
Each run uses a short PID-scoped tmpfs path, a roughly 256 KiB mutable symlink
view, a 2 GiB scratch ceiling, two-second growth checks, bounded logs, and
post-run cleanup. P3 remains correctness and lifecycle evidence only; P4/P5
service and performance qualification has not begun.
