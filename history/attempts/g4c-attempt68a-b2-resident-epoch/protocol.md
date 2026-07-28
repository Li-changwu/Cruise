# G4c Attempt 68a: B=2 Device-Resident Generation Epoch

Date frozen: 2026-07-24

## Question

Can one Device UDF transaction repeatedly invoke the accepted true B=2 AIR,
perform per-request greedy argmax and EOS control, and advance two independent
Paged-KV states without a Host round trip between decoder steps?

## Frozen mechanism

- Every iteration invokes the single Attempt 67b B=2 AIR exactly once. The UDF
  does not invoke two B=1 graphs.
- The decoder ABI is token `[2,1]`, position `[2]`, sequence length `[2,1]`,
  key/value cache `[28,4,128,4,128]`, slot mapping `[2]`, active mask `[2]`,
  block table `[2,2]`, and explicit tiling `[72]`.
- Each request owns two disjoint physical blocks under block table
  `[[1,0],[3,2]]`.
- Control is `[max_steps,eos_0,eos_1,sampling_mode,graph_variant]`; only greedy
  mode 0 and graph variant 0 are accepted.
- The UDF returns padded per-step logits and tokens, complete final key/value
  cache, and every final per-request token/position/length/slot/active field.

## Cases

- `K=1,2,4` with both requests active at heterogeneous starting positions 0
  and 2.
- `K=8` with both requests active at position 0, staying inside the frozen
  logical capacity of eight positions.
- Active plus empty, and already-finished plus active.
- Controlled independent EOS: request 0 uses its real first generated token as
  EOS and stops after one step; request 1 retains configured Qwen EOS `151645`
  and continues to K=4. This branch is labelled and is not a claim of natural
  configured-EOS occurrence.

## Pass rules

- Host batched graph loop and Device UDF match at `rtol=5e-3, atol=5e-3` for
  every active per-step logits tensor and complete final Paged KV.
- Every generated token is the greedy argmax of its real request logits.
- Token histories and all final token/position/length/slot/active fields match
  exactly.
- Empty, already-finished, and newly EOS-finished slots remain frozen on all
  later iterations; inactive request cache blocks and all unaddressed cache
  elements remain elementwise exact.
- K=1 Host and Device results preserve Attempt 67a B=2 eager continuity.
- Host submits the B=2 graph once per model call. Device uses one Feed and one
  Fetch for the complete batch epoch.
- NPU 7 is idle before Host, between routes, and after Device.

## Claim boundary

Passing closes only the B=2 resident-epoch correctness and Host-call-count
sub-gate. B=4, stable performance benefit, recovery, and vLLM-Ascend scheduler
integration remain open.
