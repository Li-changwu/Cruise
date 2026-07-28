# G4c Attempt 67e-r1: Corrected B=2 Native GE Acceptance

Date frozen: 2026-07-24

Attempt 67e executed one static B=2 graph for three semantic cases. All native
outputs passed the eager comparator, but its final launch checker incorrectly
multiplied static graph metadata counts by the three `RunGraph` calls. GE emits
the inspected kernel metadata once for the built graph.

This is a post-hoc acceptance over immutable Attempt 67e artifacts. It does not
rerun the native graph and does not modify the original Attempt 67e raw path.

## Pass Rules

- The frozen comparator result SHA256 is
  `35d833da9b647a02c8ea58732979eb3d5ad255a174487a1d4787358a92e61858`
  and reports `pass=true` for all three cases.
- The frozen native output SHA256 is
  `808c1a3c2a9edff951fa96cbaeda6ea638fbdab063a06f823e7a768132f26530`.
- Native execution and comparison statuses are zero.
- Static graph metadata contains 56 ExactQk, 28 Bf16Barrier, 56
  Bf16Materialize, and 197 decoder linear entries.
- Each linear operator has exactly one metadata entry, uses MatMulV2, and all
  residual projections remain defused.
- The original before/after evidence and the acceptance-time checks both show
  NPU 7 idle.

## Claim Boundary

Passing closes only B=2 single-step native GE semantics. It does not close the
B=2 resident Device UDF epoch, independent EOS, B=4, performance, recovery, or
vLLM integration.
