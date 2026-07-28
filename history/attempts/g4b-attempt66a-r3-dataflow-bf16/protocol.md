# G4b Attempt 66a-r3: genuine-BF16 DataFlow full-decoder smoke

## Question

Can DataFlow load the clean Attempt 65 AIR, accept its BF16 Paged-KV inputs,
and execute one complete decoder step with the same four-output semantics?

CANN 9.0 exposes `DT_BF16` in the C++ and Torch-plugin paths but omits it from
the ordinary Python NumPy dtype maps. NumPy 1.26.4 in this environment has no
built-in BF16 scalar dtype. This harness uses a project-local
`ml_dtypes.bfloat16` package and registers that genuine dtype as `DT_BF16`
inside its own process. It does not patch the installed toolkit or Conda
environment.

Attempt 66a is preserved as a pre-DataFlow dependency failure. Attempts
66a-r1/r2 reached GE compilation. The r2 log proves the authoritative AIR ABI
order is token, position, sequence length, key cache, slot mapping, block
table, value cache and tiling: GE accepted the first three inputs, then
rejected the first BF16 cache input as `DT_UNDEFINED`. Attempt 66a-r3 changes
only the BF16 representation from a `uint16` alias to the genuine
`ml_dtypes.bfloat16` dtype.

The cache NPZ stores raw BF16 bits as `uint16`. The harness reinterprets those
bits as `ml_dtypes.bfloat16` without numeric conversion. `DT_UINT16` remains a
separate registered dtype.

## Single variable

Replace the native GE Session wrapper with one Host-side DataFlow
`GraphProcessPoint`. Do not add a Device UDF, loop, argmax, EOS or state update.

## Pass conditions

- Physical NPU 7 is idle before and after execution.
- The DataFlow graph accepts all 8 inputs, including BF16 key/value caches.
- It returns exactly logits, key cache, value cache and next position.
- All outputs pass the frozen Attempt 65 eager comparison at
  `rtol=5e-3, atol=5e-3`.

Passing proves only that the accepted G4a artifact is embeddable in DataFlow.
