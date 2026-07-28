# G4 Device-Resident Generation Kernel Protocol

Date frozen: 2026-07-22

## Total Goal

Build a semantically correct, measurable, and safely recoverable fixed-shape
generation kernel before integrating with the vLLM-Ascend scheduler. Host keeps
admission, global fairness, allocation, cancellation, and recovery. Device owns
only a bounded latency-critical generation epoch.

## Frozen Model Continuity

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
- Hardware: one idle Ascend 910B2, physical NPU 7.
- CANN: 9.0.0; torch/torch_npu: 2.9.0.
- Tensor parallelism: 1.
- Decode graph shapes are static within a graph variant.

The complete G4 gate cannot be claimed with the available 14B checkpoint as a
silent substitute. A 14B run may be labelled only as a development route.

## G4a: Fixed B=1 Complete Decoder Step

Host provides one token, position, block table, slot mapping/sequence length,
and initialized per-layer Paged KV storage. The graph executes embedding, all
Qwen decoder layers, final norm and LM head, returning logits and the updated
Paged KV state.

Correctness requirements:

- eager and graph use identical weights, initial Paged KV, block allocation,
  token, position, dtype and precision configuration;
- logits are finite, have identical shape, pass `rtol=5e-3, atol=5e-3`, and
  produce the same greedy token;
- the written KV slot passes `rtol=5e-3, atol=5e-3` for every layer;
- all KV bytes outside the addressed slot remain elementwise unchanged;
- position, block-table and slot-mapping interpretation is identical;
- all registered positions/capacity cases pass, not only a selected sample.

The QK boundary is a prerequisite. The accepted lowering is raw BF16 QK,
explicit FP32 scaling, then a materialized BF16 tensor. The legacy BF16
`RealDiv` route is not accepted because Attempts 50-51 isolate sparse errors.

## G4b: B=1 Device-Resident Real Generation Epoch

A Device UDF invokes the correct G4a graph for `K in {1,2,4,8}`. Each iteration
performs real greedy argmax, EOS comparison, position/slot progression and
Paged KV mutation on device. Host performs one Feed and one Fetch for the whole
epoch.

For every K, Host-loop and Device-UDF token sequence, per-step logits, executed
step count, finish reason, final position and final Paged KV must satisfy the
G4a correctness rules. EOS and maximum-step termination are both mandatory.

## G4c: Fixed B=2/4 Control

Extend the same bounded epoch to fixed B=2 and B=4 with an active mask,
different starting lengths, independent EOS, and empty slots. Finished or
empty slots must not mutate token, length or Paged KV state. Request insertion,
preemption and continuous batching remain outside G4.

## Final Performance and Recovery Gate

- Host model submissions change from K to one Feed/Fetch transaction.
- K >= 2 yields a stable median wall-time benefit over a warmed, semantically
  identical Host graph loop; report IQR and alternating route order.
- Report Host process CPU time and the strongest available device-gap proxy.
- Capacity overflow, invalid block/slot metadata, unavailable graph variants,
  graph execution errors and unsupported semantics are rejected before unsafe
  mutation or return an explicit fallback code.
- The Host fallback replays from a known valid state and reproduces the Host
  baseline result.

## Claim Boundary

A layer slice, synthetic token rule, same-AIR Host/Device match, or successful
ACLGraph generation does not pass G4. vLLM-Ascend scheduler integration starts
only after G4a, G4b, G4c, performance and recovery gates all pass.

