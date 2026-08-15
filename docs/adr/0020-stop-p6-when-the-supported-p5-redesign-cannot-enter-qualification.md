---
status: accepted
---

# Stop P6 when the supported P5 redesign cannot enter qualification

ADR 0017 allows one profiler-attributed P5 redesign after the initial
Graph/Owner performance miss. That redesign must first pass an exact B=4,
K=384 attention component through ordinary Graph and public GraphPp. A
component failure is not a second performance measurement, but it prevents the
only permitted redesign from entering the second pair or six-start matrix.

Two bounded public compute-plane paths were exhausted on CANN 9.0 and Ascend
910B2. The public ACL path built and exactly executed a self-contained FIA OM,
but public DataFlow exposes no process point that accepts the OM. The public
custom-OPP path generated a valid 31-input FIA AIR and installed the official
ops-transformer v9.0.0 ordinary and relocatable objects in an isolated vendor
tree. Ordinary Graph and GraphPp both ignored the static object, fell back to
online `te_fusedinferattentionscore_*` compilation, and failed before execution.

P5 is therefore Stopped / Unqualified on this stack. The second Graph/Owner
pair and six-start matrix are not run, and P6 remains closed. This decision does
not lower or reinterpret any ADR 0017 performance threshold.

Reopening P5 requires a newly documented public precompiled-model process
point, a vendor-supported FIA static-package path that passes the exact mini
Graph/GraphPp gate, or a different supported Device compute plane. Reopening
also requires a new ADR, pinned source and package identities, and fresh
target-NPU evidence.
