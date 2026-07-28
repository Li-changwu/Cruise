# G4b Attempt 66a: DataFlow full-decoder smoke

## Question

Can DataFlow load the clean Attempt 65 AIR, accept its BF16 Paged-KV inputs,
and execute one complete decoder step with the same four-output semantics?

CANN 9.0 exposes `DT_BF16` in the C++ and Torch-plugin paths but omits it from
the ordinary Python NumPy dtype maps. The harness registers the existing BF16
enum with `ml_dtypes.bfloat16` inside its own process; it does not patch the
installed toolkit.

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
