# G4c Attempt 69d: B=4 Native GE Correctness

Date frozen: 2026-07-24

## Prerequisites

- Attempt 69b B=4 eager reference SHA256:
  `a7d65e455a77a561352a8f3796d94ec86e1e429ebe942feacd5b14013123fdd8`.
- Attempt 69c B=4 AIR SHA256:
  `de4a7bf439337970b343eb1fa91c3dd326545e81a88c94680c355234f96044bb`.
- Attempt 69c-r1 ABI, graph, and deduplicated-export acceptance is valid.
- Attempt 69a installs the B=1/B=2/B=4 element-count `Bf16Barrier`.

## Question

Does native GE execution of the single static B=4 AIR reproduce the accepted
B=4 eager results for all-active heterogeneous lengths, two active plus two
empty slots, and finished/active/empty/active slots?

## Pass Rules

- Full logits and complete key/value Paged KV match Attempt 69b at
  `rtol=5e-3, atol=5e-3` for all three cases.
- Active-request greedy tokens match and active logits are finite.
- Next position matches exactly and increments only for active requests.
- Every KV element outside active addressed slots remains exactly unchanged.
- Every inactive request's two-block cache slice remains exactly unchanged.
- Static native graph launch metadata contains exactly 112 `ExactQk`, 28
  `Bf16Barrier`, 56 `Bf16Materialize`, and 197 decoder linear entries. These
  counts describe one compiled graph and are not multiplied by three
  `RunGraph` calls.
- NPU 7 is idle before and after the run.

## Claim Boundary

This attempt closes only B=4 single-step native GE semantics. It does not
close the B=4 resident epoch, performance, recovery, or vLLM integration.
