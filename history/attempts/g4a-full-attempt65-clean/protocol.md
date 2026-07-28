# G4a Attempt 65: clean residual-defused decoder

## Claim

A four-output, fixed-B=1 Qwen2.5-7B decoder graph can preserve eager logits,
Paged-KV and position semantics when all 56 projection-residual epilogues are
explicitly defused.

## Single variable from Attempt 64a

Remove only the diagnostic `layer_hiddens` collection and fifth graph output.
Keep the 8 inputs, ExactQk, score materialization, all 197 MatMulV2 contracts,
and the 56 Bf16Materialize nodes before residual additions.

## Pass conditions

- Eager reference is byte-identical to clean Attempt 53k.
- AIR ABI is exactly 8 inputs and 4 outputs: logits, key cache, value cache,
  and next position.
- AIR contains MatMulV2=197, ExactQk=28, Bf16Barrier=28,
  Bf16Materialize=56 and BatchMatMul=29, with MatMul=0.
- Four recurrent native steps pass logits and full Paged-KV comparison at
  rtol=5e-3 and atol=5e-3, preserve every unaddressed KV slot exactly, match
  greedy tokens and positions, and produce only finite logits.
- Runtime launches contain 28 ExactQk, 28 score barriers, 56 materializations,
  197 distinct MatMulV2 kernels, and all 56 o_proj/down_proj launches remain
  defused.

Passing closes G4a only. Device-side argmax, EOS and resident epoch control are
G4b.
