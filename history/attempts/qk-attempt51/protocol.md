# G4a QK Attempt 51: Materialized BF16 Scaling Boundary

Date: 2026-07-22

## Question

Can explicit FP32 scaling followed by a materialized BF16 graph output recover
the frozen eager QK semantics after Attempt 50 proved the raw `ExactQk` output
is elementwise exact?

## Controlled Change

All hardware, operands, tiling bytes, custom-op package, GE precision mode, and
references remain those of Attempt 50. The only change is that the three scaled
outputs remain BF16 at the AIR boundary. This prevents GE from treating an
adjacent `FP32 -> BF16 -> FP32` pair as a removable conversion.

The graph returns raw BF16 QK, legacy BF16 division, explicit FP32 division
rounded to BF16, and explicit FP32 reciprocal multiplication rounded to BF16.

## Pass Rule

The attempt passes only if raw QK remains elementwise exact at all four
positions and at least one explicit scaling candidate is elementwise exact at
all four positions versus eager. A pass clears only the frozen QK/scaling
boundary; it does not claim full-decoder correctness.

