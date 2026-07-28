# G4b Attempt 66a-r2: DataFlow full-decoder smoke

## Question

Can DataFlow load the clean Attempt 65 AIR, accept its BF16 Paged-KV inputs,
and execute one complete decoder step with the same four-output semantics?

CANN 9.0 exposes `DT_BF16` in the C++ and Torch-plugin paths but omits it from
the ordinary Python NumPy dtype maps. NumPy 1.26.4 in this environment has no
BF16 scalar dtype. The harness therefore registers `uint16` as a bit-preserving
container for `DT_BF16` inside its own process and removes the competing
`DT_UINT16` registration. This experiment uses no genuine UINT16 tensor and
does not patch the installed toolkit.

Attempt 66a is preserved as a pre-DataFlow dependency failure because
`ml_dtypes` was unavailable. Attempt 66a-r1 established that the process-local
BF16 bridge reached DataFlow compilation, then failed because it used Python
`forward` argument order. Attempt 66a-r2 changes only the input order to the
authoritative AIR ABI: token, position, sequence length, key cache, slot
mapping, block table, value cache and tiling.

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
