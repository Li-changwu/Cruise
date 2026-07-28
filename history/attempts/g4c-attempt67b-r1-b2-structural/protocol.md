# G4c Attempt 67b-r1: Corrected B=2 AIR Structural Audit

Date frozen: 2026-07-24

Attempt 67b successfully generated the AIR but its checker incorrectly assumed
that TorchAir internal `arg` ordinals equal Python argument ordinals. The
original export and failed checker outputs remain unchanged.

This validation uses the immutable Attempt 67b graph and distinguishes the two
INT32 `[2]` inputs from their frozen dataflow:

- slot mapping has 112 direct consumers: two requests times key/value updates
  times 28 layers;
- active mask has those 112 update consumers plus the final next-position
  freeze, for 113 direct consumers.

All ABI shapes, output shapes, linear contracts, and six predeclared operator
counts from Attempt 67b remain mandatory. This attempt changes only the invalid
semantic-identification rule; it does not rebuild or rewrite the AIR.

