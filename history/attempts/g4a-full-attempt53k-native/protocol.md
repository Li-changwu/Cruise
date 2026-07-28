# G4a Attempt 53k Native Complete-Decoder Gate

Date frozen: 2026-07-23

## Claim

The exported Attempt 53k AIR executes four recurrent complete Qwen2.5-7B
decoder steps under native GE on idle physical NPU 7. Each step consumes the
previous native output KV cache and next position rather than reloading the
eager state.

## Frozen Inputs And Outputs

- AIR: `qwen_full_decoder_step_attempt53k.air` with 342 external weight files.
- Precision mode: `must_keep_origin_dtype`.
- Inputs: token, position, sequence length, key cache, slot mapping, block
  table, value cache and explicit ExactQk tiling.
- Outputs: FP32 logits, BF16 key cache, BF16 value cache and next position.
- Four prompt tokens and initial cache are taken from the immutable 53k eager
  reference.

## Pass Conditions

For every step:

- logits, the addressed K slot and the addressed V slot are within
  `rtol=5e-3, atol=5e-3` of the 53k eager reference;
- greedy token and next position are identical;
- every unaddressed K/V element is bitwise unchanged from the preceding native
  input;
- complete K/V outputs remain within the frozen tolerance;
- runtime logs contain actual `te_exactqk` and `te_bf16barrier` launches;
- physical NPU 7 is empty before and after the run.

## Boundary

Passing this gate establishes G4a only. Sampling, EOS checks and recurrent
control still execute on Host and must move into the Device UDF in G4b.
