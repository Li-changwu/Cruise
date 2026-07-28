# G2g Attempt 40 Result

## Verdict

`SetScheduleMode(2)` is not a valid repair for the native GE/DataFlow path in
the installed CANN 9.0.0 runtime. The online compiler accepted the setting, but
model loading rejected it before any `ExactQk` kernel launch.

## Direct Evidence

- Artifact integrity, custom OPP setup, and native Host compilation passed.
- Fresh online compilation succeeded and emitted an AIC object.
- Compiler JSON recorded `schedule_mode=2`, `blockDim=24`, and `opParaSize=80`.
- `aclrtLaunchKernelV2` rejected `schedMode=2` with Runtime error `107000`:
  `Expected value: [0, 2)`.
- The first `RunGraph` returned `4294967295`; no output tensor was produced.
- No process remained on physical NPU 7 after exit.

## Interpretation

The failure is earlier and different from the AIC/AIV result 145 observed for
schedule modes 0 and 1. Mode 2 is accepted by the direct PyTorch custom-op
launch used in the numerical screen, but it is outside the legal range of the
`aclrtLaunchKernelV2` path selected by native GE/DataFlow model deployment.
This falsifies schedule mode 2 as a sufficient GE integration repair without
making a claim about the original QK computation body.

## Preserved Artifacts

- Remote protocol: `/root/ascend-control-g2g-20260719/attempt40-protocol.md`
- Remote raw results: `/root/ascend-control-g2g-20260719/raw-attempt40/`
- Online compiler cache: `/root/ascend-control-g2g-20260719/cache-attempt40/`
