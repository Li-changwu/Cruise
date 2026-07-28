# G2g Attempt 43: Offline Completion Of Attempt 42

## Question

Do the four already-produced Attempt 42 outputs pass the frozen eager
comparison after correcting only the launch-metadata parser, without rerunning
or otherwise touching the NPU?

## Why A Separate Attempt Is Required

Attempt 42 completed native execution with status 0 and emitted four 896-byte
outputs. Its harness then stopped because the grep expression required
`te_exactqk` to appear before `kernelType=0`, whereas the one preserved launch
line records `kernelType=0` before the kernel name. This is a secondary
instrumentation-order bug. The numerical comparator was never invoked.

## Frozen Inputs

- Attempt 42 `native.stdout.log` SHA-256:
  `3782966d322afe7b1cd12021784e43fa4c9f140688207855fce6e663355d7ac0`.
- Attempt 42 `status.tsv` SHA-256:
  `28ec6236657de1162a91b92e5edc05469e78421a576940845dc8a8fd3a9a4f9b`.
- Preserved launch-line file SHA-256:
  `41064632f9908c06f84bbb36f4025dde6e34fb308ff59a39c3c6cb733b315f44`.
- Four-output tree SHA-256:
  `8dce013bc213928ecea41eddf6d6dedc4db4bf9785cbd7afd6ce7970670c7d85`.
- Output names are exactly `step{1..4}_qk_scores.bin`, each 896 bytes.
- Comparator and eager reference remain byte-identical to Attempt 42.
- Correctness remains exact equality plus `rtol=5e-3`, `atol=5e-3`.

## Decision Rules

Attempt 43 does not initialize CANN, query `npu-smi`, or execute an NPU binary.
It passes only if all frozen hashes match, Attempt 42 records native status 0,
the single launch line independently contains `te_exactqk`, `kernelType=0`,
`coreDim=24`, and `schemMode=1`, the four-file contract matches, and the
unchanged comparator reports all four positions elementwise exact.

A pass completes the minimal GE/AIR embeddability and numerical-fidelity gate
for the static-tiling fixed-shape QK. It does not expand the claim to full Qwen,
Device UDF recurrence, or vLLM.
