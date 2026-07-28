# G4a Attempt 55c: Disable TbeMultiOutputFusionPass

## Question

Does GE's `TbeMultiOutputFusionPass` cause the first layer-0 BF16 mismatch by
combining the AIR `Swish` and `Mul_12` nodes into a two-output `SwishMul_12`
kernel?

## Controlled Change

Reuse the immutable Attempt 55a AIR, eager reference, ABI, inputs, native host
logic, precision mode, and custom operators. The only runtime change relative
to 55a is a `ge.fusionSwitchFile` that disables `TbeMultiOutputFusionPass`.
`TbeEltwiseFusionPass` remains at its default setting.

Attempt 55b established that disabling `TbeEltwiseFusionPass` alone leaves the
fused kernel and every output byte unchanged. Its log localized the successful
`Swish`/`Mul_12` match to `TbeMultiOutputFusionPass`.

New immutable source, raw-result, and cache directories are used. Compilation
cache and `RESOURCE_CONFIG_PATH` remain disabled as in Attempt 55a.

## Pass Rule

- GE accepts the fusion switch and runs all four frozen steps on physical NPU 7.
- Neither `Start buffer fusion: TbeMultiOutputFusionPass` nor
  `te_fused_op_swish_mul` appears in the native log.
- All 18 outputs remain finite and within `rtol=5e-3, atol=5e-3`.
- Gate/product/down-projection mismatch counts are compared directly with 55a;
  only a reproducible reduction supports the fusion-causality hypothesis.
- ExactQk and Bf16Barrier launches are present, and NPU 7 is idle before and
  after the experiment.

This is a layer-0 single-variable diagnostic. It cannot pass complete G4a.
