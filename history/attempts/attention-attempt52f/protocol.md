# G4a Attempt 52f: Attention Boundary Localization

Date: 2026-07-23

## Question

Where do the remaining 1/5 FP16 Attention mismatches first appear after
Attempt 52e proved that a second softmax-output barrier does not remove them?

## Controlled Change

The model, revision, four hidden rows, initial caches, positions, weights,
precision mode, ExactQk kernel, two barriers and all 16 frozen outputs remain
those of Attempt 52e. The graph only gains observable outputs for:

1. masked BF16 scores;
2. barrier-preserved BF16 probabilities;
3. BF16 P x V output;
4. flattened BF16 Attention value;
5. BF16 output-projection result before its FP16 presentation cast.

The 72-byte tiling tensor remains an explicit graph input. Outputs 17-21 are
diagnostic only and do not change the computation consumed by Attention.

## Pass Rule

This diagnostic run is valid only if:

- its eager route reproduces all 16 frozen Attempt-7 outputs elementwise;
- native GE emits all 22 outputs with the declared ABI and within tolerance;
- the first non-exact diagnostic tensor can be identified for each step;
- the observed AIR input/output ABI matches the generated manifest;
- ExactQk and exactly two Bf16Barrier nodes are present in the AIR;
- the native runtime records at least two Bf16Barrier launches;
- the run uses idle physical NPU 7 before and after execution.

This is still a layer-slice prerequisite. It does not pass G4a without every
decoder layer, logits and Paged KV.
