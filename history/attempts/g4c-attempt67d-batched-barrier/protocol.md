# G4c Attempt 67d: B=1/B=2 BF16 Barrier Contract

Date frozen: 2026-07-24

Attempt 67c reached the first decoder barrier with input
`[2, 28, 1, 8]` BF16, but the installed Attempt 54 barrier accepted only
`[1, 28, 1, 8]` and its kernel copied a hard-coded 224 elements.

This attempt preserves the `Bf16Barrier` op name and bitwise identity contract,
while allowing only batch one or two. Host tiling records the actual 224 or 448
element count; the device kernel copies exactly that count through the same
scalar materialization boundary. The upper bound remains 448 elements.

Passing this build creates a fresh installed custom-op package. Build success
alone does not close correctness; Attempt 67e must rerun all three full-decoder
native cases with this package and reproduce the 67a reference.

