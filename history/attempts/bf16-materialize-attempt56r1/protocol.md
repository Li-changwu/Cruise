# Attempt 56r1: General BF16 Materialization Operator

## Question

Can an opaque device kernel materialize any contiguous BF16 tensor needed by
the Qwen decoder boundary probes, including the `[1, 1, 18944]` MLP gate, while
preserving every BF16 bit?

## Mechanism

`Bf16Materialize` obtains the element count from GE tiling data and copies the
input through an AIV UB buffer. It accepts positive contiguous BF16 tensors up
to 32768 elements. A scalar tail path handles counts not divisible by 16.

The operator is built in a new copy of the custom-op project and installed in a
new path. It does not replace the frozen Attempt 54 `Bf16Barrier` package.

Attempt 56 failed at compile time because CANN 9.0.0 declares
`SaveToBuffer` with a `void` return type. This revision changes only that API
call and all artifact paths; the operator, input, and pass rule are unchanged.

## Preflight Pass Rule

- The eager custom-op implementation is bitwise identity at shape
  `[1, 1, 18944]`.
- The AIR contains exactly one `Bf16Materialize` node.
- Native GE output is bitwise identical to the input.
- The native log contains an actual `te_bf16materialize` launch.
- Physical NPU 7 is idle before and after the probe.

Only after this preflight passes may the operator be inserted between layer-0
Swish and MLP multiplication.
