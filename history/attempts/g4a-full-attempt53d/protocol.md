# G4a Attempt 53d: Barrier-Protected Fixed-B=1 Complete Qwen Decoder Step

Date frozen: 2026-07-23

## Claim

This attempt tests one complete Qwen2.5-7B-Instruct decoder step. It is not a
layer slice and it does not yet claim a device-resident generation epoch.

The graph consumes one token, its position, the current sequence length, a
vLLM-Ascend-compatible block table and slot mapping, all 28 layers of Paged KV,
and the explicit ExactQk tiling tensor. It executes embedding, every decoder
layer, final RMSNorm and LM head, and returns logits, updated Paged KV and the
next position.

## Frozen Model And Cache Variant

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
- Full checkpoint: all 339 tensors, without layer substitution.
- Batch size: 1.
- Physical KV layout per K or V tensor:
  `[28, 2, 128, 4, 128]` BF16.
- Logical attention capacity of this first static graph variant: 8 tokens.
- Block table: `[1, 0]`; logical block zero therefore maps to physical block
  one and is not an identity mapping.
- Slot mapping for position `p`: `1 * 128 + p`.
- QK lowering: raw BF16 `ExactQk`, FP32 scale, BF16 cast, then the opaque
  `Bf16Barrier` proven by Attempt 54b and the Attempt 52d-f sequence.

The physical page size is the 128-token size reported by the current
vLLM-Ascend attention backend. The first registered graph deliberately has an
8-token logical operating envelope so the already validated ExactQk kernel can
be reused without changing two variables at once. Position 8 is a capacity
failure for this graph variant and must route to Host fallback; it is not
silently truncated.

## Independent Eager Screen

Before AIR export, run four frozen prompt tokens through both:

1. Hugging Face Qwen eager attention with `DynamicCache`;
2. the complete manual Paged-KV decoder step used for AIR export.

For each step:

- logits must be finite and within `rtol=5e-3, atol=5e-3`;
- greedy token must be identical;
- the newly written K and V slot must be within the same tolerance for every
  layer;
- every byte outside the addressed slot must remain elementwise unchanged.

## Native AIR Gate

Native GE must reproduce all four manual eager steps under
`must_keep_origin_dtype`. Logits and every written layer slot use the frozen
tolerance; greedy tokens must match; all unaddressed Paged-KV bytes must remain
elementwise unchanged. The AIR must launch `te_exactqk` and must use idle
physical NPU 7. The AIR must also contain 28 `Bf16Barrier` nodes and launch
the corresponding barrier kernel tasks.

## Boundary

Passing Attempt 53 establishes only G4a. Greedy sampling, EOS, recurrent
device-side control, B=2/4 masks and vLLM scheduler integration remain G4b/G4c.
