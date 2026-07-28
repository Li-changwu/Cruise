# G4c Attempt 67b: B=2 AIR Export and Structural Audit

Date frozen: 2026-07-24

## Prerequisite

Attempt 67a reference SHA256:
`be494220c90f2f0cbec848a5a8ce81ab889007d59f0f116dbfe44b18ce9cd3ce`.

## Question

Can the accepted `B=2` eager implementation be represented by one static AIR
with the frozen nine-input ABI, including a live active-mask dependency?

## Pass Rules

- AIR and graph protobuf are generated from the frozen 67a reference.
- The ABI contains nine contiguous Data inputs with argument-ordinal semantics
  and four outputs with the frozen B=2 shapes and dtypes.
- All 197 full-decoder linear operators remain `MatMulV2` with transposed
  external weights; their activation leading dimension is two.
- The graph contains 56 `ExactQk`, 28 BF16 scale barriers, 56 residual/MLP
  materialization boundaries, and 28 attention batch matmuls.
- Exactly one active-mask Data node exists and has at least one graph consumer.
- NPU 7 is idle before and after export.

## Claim Boundary

This structural gate does not claim native GE numerical correctness, recurrent
generation, Device UDF execution, B=4, performance, recovery, or vLLM
integration.

