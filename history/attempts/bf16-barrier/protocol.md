# Attempt 54: Opaque BF16 Materialization Barrier

## Question

Can an opaque device kernel preserve the scale-to-BF16 rounding boundary that
GE removes when the BF16 value is immediately consumed as FP32?

## Controlled Change

The new `Bf16Barrier` custom op has one BF16 input and one BF16 output with an
identical shape. Its AIV kernel copies exactly 448 raw 16-bit values through UB.
It performs no arithmetic and therefore must be bitwise identical.

The operator is built in a copy of the custom-op project and installed beside,
not over, the frozen Attempt-47 `ExactQk` package.

## Pass Rule

- direct eager semantics are identity;
- native GE output is elementwise identical for all four frozen scaled-QK
  tensors;
- the native log contains a `bf16_barrier` launch;
- physical NPU 7 is idle before and after the probe.

Only after this probe passes may the barrier be inserted between QK scaling
and FP32 softmax in the complete Attention slice.

