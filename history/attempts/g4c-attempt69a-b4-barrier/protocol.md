# G4c Attempt 69a: B=1/B=2/B=4 BF16 Barrier Contract

Date frozen: 2026-07-24

Attempt 67d closed B=1/B=2 by replacing a hard-coded 224-element copy with
runtime element-count tiling. G4c now requires the same identity barrier for a
true B=4 decoder input `[4, 28, 1, 8]` BF16.

This attempt preserves the `Bf16Barrier` op name and bitwise identity contract,
while allowing only batch one, two, or four. Host tiling records the actual
224, 448, or 896 element count; the device kernel copies exactly that count
through the same scalar materialization boundary. The upper bound is 896
elements.

Passing this build creates a fresh installed custom-op package. Build success
alone does not close B=4 correctness; the subsequent true B=4 eager, AIR, and
native experiments must reproduce independent B=1 request oracles.
