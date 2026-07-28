# G4c Attempt 69d-r1: B=4 Native GE after AIR Relocation

Date frozen: 2026-07-24

## Delta from Attempt 69d

Attempt 69d failed before its first graph execution because the Attempt 69c AIR
retained deleted temporary external-weight paths. Attempt 69c-r2 reserialized
the same graph with all 342 paths targeting the persistent deduplicated export.

This attempt reuses the immutable Attempt 69d runner and validators. The only
execution-artifact change is:

- old AIR: Attempt 69c, SHA256
  `de4a7bf439337970b343eb1fa91c3dd326545e81a88c94680c355234f96044bb`;
- new AIR: Attempt 69c-r2, SHA256
  `263b2acf291e13f6a84042ded53c8dccabb1fa847dcdcbbbe0ece418610ad1e3`.

## Pass Rules

- All three B=4 cases execute through native GE.
- Full logits and complete key/value Paged KV match Attempt 69b at
  `rtol=5e-3, atol=5e-3`.
- Active greedy tokens, next position, inactive cache slices, and all
  unaddressed KV elements satisfy the frozen Attempt 69d rules.
- Static graph launch metadata contains exactly 112 `ExactQk`, 28
  `Bf16Barrier`, 56 `Bf16Materialize`, and 197 decoder linear entries.
- Attempt 69c-r2 relocation evidence is valid, and NPU 7 is idle before and
  after execution.

## Claim Boundary

This closes only B=4 single-step native GE semantics. B=4 Device UDF resident
epochs, stable performance, recovery, and vLLM integration remain open.
