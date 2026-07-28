# G4b Attempt 66b-r1: B=1 device-resident real generation epoch

## Claim

For `K in {1,2,4,8}`, one Device UDF transaction can repeatedly invoke the
accepted Attempt 65 decoder, perform deterministic greedy argmax over its real
FP32 logits, compare the generated token with EOS, advance position/sequence
length/slot mapping, and carry real Paged-KV state without returning to Host.

## Contract

The first eight inputs are the Attempt 65 decoder ABI. Input nine is
`[max_steps, eos_token, sampling_mode, graph_variant]`, where only greedy mode
0 and graph variant 0 are supported in this gate.

The UDF returns padded per-step logits, generated token history, final key and
value caches, final position, and a control record containing executed steps,
finish reason, status, final token/position/length, model-call count and a
fallback flag. Any rejected epoch returns the original KV/position boundary.

## Pass conditions

- Host and Device routes match exactly for token history, per-step logits,
  final Paged-KV, position and control for K=1/2/4/8.
- Argmax is computed from every real decoder logits tensor on device; no
  synthetic token or `eos_after_step` input exists.
- Configured Qwen EOS `151645` is checked on every iteration. A separate
  controlled branch may set the runtime EOS ID to an actually generated token
  to exercise early termination; it must be labelled and cannot be presented
  as natural configured-EOS occurrence.
- Device route uses one Feed and one Fetch per epoch.
- K>=2 performance is measured only after correctness passes.

G4c batching and vLLM integration remain outside this attempt.

The Host baseline uses the same accepted AIR through native `Session::RunGraph`
and feeds each generated greedy token and returned Paged-KV into the next step.
It does not use the remaining prompt tokens from the four-step G4a reference.
The Device route wraps the AIR as an invoked GraphPp closure under the FunctionPp
UDF, which is required because Attempt 66a-r5 isolated BF16 failure on the
direct Host-to-GraphPp route while Attempt 66a-r6-r2 proved native BF16
Host-to-Device-UDF transport and bitwise round-trip correctness.
