# G4c Attempt 69c: B=4 AIR Export and Structural Audit

Date frozen: 2026-07-24

## Prerequisite

Attempt 69b reference SHA256:
`a7d65e455a77a561352a8f3796d94ec86e1e429ebe942feacd5b14013123fdd8`.

## Question

Can the accepted `B=4` eager implementation be represented by one static AIR
with the frozen nine-input ABI, including a live active-mask dependency?

## Pass Rules

- AIR and graph protobuf are generated from the frozen 69b reference.
- The ABI contains nine contiguous Data inputs with argument-ordinal semantics
  and four outputs with the frozen B=4 shapes and dtypes.
- All 197 full-decoder linear operators remain `MatMulV2` with transposed
  external weights; their activation leading dimension is four.
- The graph contains 112 `ExactQk`, 28 BF16 scale barriers, 56 residual/MLP
  materialization boundaries, and 28 attention batch matmuls.
- Exactly one active-mask Data node exists and has at least one graph consumer.
- The temporary 15 GB export is materialized persistently through a
  hash-verified dedup manifest. Files identical to the accepted B=2 export are
  hard-linked; B=4-unique external files are copied.
- NPU 7 is idle before and after export.

## Claim Boundary

This structural gate does not claim native GE numerical correctness, recurrent
generation, Device UDF execution, performance, recovery, or vLLM integration.
