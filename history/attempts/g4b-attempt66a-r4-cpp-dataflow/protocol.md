# G4b Attempt 66a-r4: C++ DataFlow full-decoder smoke

## Question

Can the official C++ DataFlow Host API embed the accepted Attempt 65 decoder
AIR and execute one semantically correct step when BF16 Paged-KV inputs are
represented directly by `ge::TensorDesc(..., DT_BF16)`?

Attempts 66a-r1/r2 showed that a `uint16` NumPy alias reaches GE as
`DT_UNDEFINED`. Attempt 66a-r3 showed that the CANN 9.0 Python extension rejects
the genuine `ml_dtypes.bfloat16` buffer code `E`. Those failures are preserved.
Attempt 66a-r4 changes only the Host DataFlow binding from Python to the
official C++ API. The AIR, ABI order, input bytes, precision mode, reference and
frozen tolerance are unchanged.

## Pass conditions

- Physical NPU 7 is idle before and after execution.
- The C++ DataFlow graph accepts all 8 inputs, including two `DT_BF16` caches.
- One Feed and one Fetch return exactly FP32 logits, BF16 key/value caches and
  the INT64 next position.
- All four outputs pass the Attempt 65 eager reference at
  `rtol=5e-3, atol=5e-3`.

Passing proves only that the accepted G4a artifact is embeddable in DataFlow.
Device UDF recurrence, greedy sampling and EOS remain G4b work.
