---
status: accepted
---

# Preserve Device control with a graph family

The first P5 Graph/Owner pair proved that the Persistent Device Model Owner can
reduce Host CPU per output token, but its correctness-first Decoder graph lost
the strongest Graph route's compute structure. It serialized the 128-token
Prefill through one-token invocations and did not preserve fused attention,
merged projections, or an in-place Paged-KV boundary. Returning token control
to the Host would discard the validated control-plane property without fixing
that compute-plane regression.

The next compute-plane candidate is a Controller-Aware Graph Family. A thin
Device Control Plane retains admission, cancellation, Output Credit, row and KV
ownership, phase transitions, and graph selection. It invokes model-load-
scoped whole-model graphs through public GraphPp closures and `RunFlowModel`.
The first fixed family contains `Prefill-B4-L128` for one full formal prompt and
`Decode-B4` for one generation step. Additional shapes or length buckets require
evidence; they are not folded into an initially dynamic universal graph.

Both graphs advance one Shared Device State Contract for the same Device KV
Lease. The contract forbids Host KV copies and full-cache graph input/output
unless an explicit supported alias contract proves that no reconstruction or
transfer occurs. The graphs preserve merged QKV and gate/up projections, native
normalization and rotary operations, fused attention, and paged KV updates.
Structural verification rejects decomposed `SoftmaxV2` or `BatchMatMul`
attention fallbacks before hardware performance work begins.

This decision does not use private `ModelPp`, per-token Host `aclmdlExecute`, a
modified system CANN, or weaker performance thresholds. ADR 0020 and the P5
Stopped / Unqualified state remain effective. Controller-Aware Graph V2 is a
new candidate, not a reinterpretation or reopening of P5. Reopening requires a
separate accepted decision after both graphs pass ordinary Graph and public
GraphPp on the pinned target, their shared KV behavior is proven, and source,
package, artifact, correctness, and copy evidence are fixed.
