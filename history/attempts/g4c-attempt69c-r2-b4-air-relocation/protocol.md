# G4c Attempt 69c-r2: Relocatable B=4 AIR

Date frozen: 2026-07-24

## Observed Packaging Failure

Attempt 69d loaded the accepted Attempt 69c AIR but failed before the first
graph execution. All 342 `FileConstant.file_path` attributes still referenced
the deleted temporary export directory:

`/dev/shm/ascend-control-g4-20260724/attempt69c-b4-export-tmp`

The deduplicated weights themselves remain valid in the persistent Attempt 69c
export. Attempt 69c-r1 proved file integrity but did not prove AIR relocation.

## Question

Can the immutable Attempt 69c graph be reserialized with every external-weight
path targeting the persistent, hash-verified export, without changing graph
node identity or type?

## Pass Rules

- Source AIR SHA256 is
  `de4a7bf439337970b343eb1fa91c3dd326545e81a88c94680c355234f96044bb`.
- Exactly 342 source `FileConstant` nodes use the frozen temporary prefix.
- Exactly 342 rewritten `FileConstant` nodes use
  `/root/ascend-control-g4-20260723/export-attempt69c-b4`.
- Every rewritten target exists and is readable.
- A reload of the new AIR has the same complete sorted `(node name, node type)`
  signature as the source graph.
- No reloaded `FileConstant` retains the old prefix.
- NPU 7 is idle before and after this host-only transformation.

## Claim Boundary

This closes only the AIR relocation defect. Native numerical correctness still
requires a fresh Attempt 69d-r1 execution.
