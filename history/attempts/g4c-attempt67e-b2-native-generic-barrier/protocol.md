# G4c Attempt 67e: B=2 Native GE with Generic Barrier

Date frozen: 2026-07-24

Attempt 67c failed before execution because the installed Attempt 54
`Bf16Barrier` accepted only B=1. Attempt 67d replaced that runtime contract
with a B=1/B=2 element-count-tiling implementation. This attempt changes only
the sourced barrier package and uses fresh source, raw, and build paths.

## Prerequisites

- Attempt 67a B=2 eager reference SHA256:
  `be494220c90f2f0cbec848a5a8ce81ab889007d59f0f116dbfe44b18ce9cd3ce`.
- Attempt 67b AIR SHA256:
  `b012da4674d62ad064ade4d89d5aac09273421315286ccc5ca305a04c3c83bf3`.
- Attempt 67b-r2 ABI and graph audits are valid.

## Question

Does native GE execution of the single static B=2 AIR reproduce the accepted
batched eager results for both-active heterogeneous lengths, active plus empty,
and finished plus active?

## Pass Rules

- Full logits and complete key/value Paged KV match 67a at
  `rtol=5e-3, atol=5e-3` for all three cases.
- Active-request greedy tokens match and logits are finite.
- Next position matches exactly and increments only for active requests.
- Every KV element outside active addressed slots remains exactly unchanged.
- Every inactive request's two-block cache slice remains exactly unchanged.
- Runtime launch evidence contains 168 ExactQk, 84 Bf16Barrier, 168
  Bf16Materialize, and 591 decoder linear launches across three graph calls.
- NPU 7 is idle before and after the run.

## Claim Boundary

This attempt closes only B=2 single-step native GE semantics. It does not close
Host/Device resident epochs, per-request EOS, B=4, performance, recovery, or
vLLM integration.
