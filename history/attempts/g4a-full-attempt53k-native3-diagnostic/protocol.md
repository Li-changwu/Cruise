# G4a Attempt 53k Native3 Layer Diagnostic

Date frozen: 2026-07-23

## Question

At which decoder layer does native3 first exceed the frozen G4a tolerance in
the newly written Paged-KV slot?

## Inputs

- Immutable Attempt 53k eager reference.
- Immutable Attempt 53k native3 inputs and four-step outputs.
- No NPU execution.

## Method

For each step and each of 28 layers, compare the 512 written K elements and
512 written V elements at physical slot `128 + step - 1`. Also verify all
native input files against the reference, preserve output hashes, check
unaddressed cache bits, and record native/eager logits top-10.

## Interpretation

- First failure at layer 0: instrument embedding, first RMSNorm, first Q/K/V
  projections and RoPE before rerunning the full graph.
- First failure after layer 0: instrument the preceding layer's attention and
  MLP boundary plus the failing layer's Q/K/V projections.
- No per-layer K/V failure but logits failure: instrument final norm/lm_head.

This diagnostic cannot pass G4a; it only chooses the next controlled probe.
