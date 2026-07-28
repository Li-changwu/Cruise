# Bounded Device-Resident Decode Control Protocol

Date: 2026-07-22

## Objective

Test whether a DataFlow Device UDF can own a bounded decode recurrence around a
real Qwen layer-0 attention/KV AIR while the Host performs one Feed and one
Fetch per transaction.

## Control Contract

The Host feeds five tensors once: hidden state, position, key cache, value
cache, and a six-element runtime control tensor:

`[max_steps, eos_token, eos_after_step, graph_switch_step, token_seed,
token_stride]`.

The Device UDF repeatedly chooses `decode_graph_0` or `decode_graph_1`, invokes
the selected model closure, feeds the returned position/K/V state into the next
iteration, derives the synthetic next token, and stops on EOS or `max_steps`.
It returns the four final model tensors plus:

`[input control..., executed_steps, final_token, finish_reason,
graph0_calls, graph1_calls, final_position]`.

Finish reason 1 is EOS and 2 is the maximum-step bound.

## Frozen Matrix

Four scenarios cover one-step execution, early EOS at steps 2 and 3,
maximum-step termination at step 4, and graph-route switches. The exported AIR
has eight KV slots but only four staged hidden rows, so its legal recurrence is
bounded to four calls from initial position zero. The Device UDF validates the
hidden/K/V capacities before entering the loop. Each route is warmed three
times and measured ten times with alternating Host/Device ordering.

## Gates

The mechanism passes only if:

1. Every final Host/Device attention, K cache, V cache, position, and control
   tensor is elementwise exact.
2. EOS and maximum-step cases report the expected step and finish reason.
3. Route counters match the frozen switch points and at least one scenario
   exercises both device route keys.
4. The device route uses one Feed and one Fetch regardless of executed steps.

Performance is reported as median wall time, Host process CPU time, speedup,
and Host CPU reduction. It is evidence for this controlled slice, not a full
decoder throughput claim.

## Claim Boundary

This prototype uses the real Qwen layer-0 attention/KV AIR, but its token rule
and EOS injection are synthetic. Both route keys currently reference the same
AIR, so the experiment validates runtime graph selection but not distinct
shape/batch-specialized graph variants. The AIR remains non-equivalent to eager
under the frozen G2d tolerance. Paged KV, logits/sampling, per-request masks,
dynamic batching, full-decoder correctness, and vLLM integration are outside
this gate.

## Preserved Negative Attempt

`raw-run2` included an eight-call case before the staged-hidden capacity check
was implemented. Control state, final position, stop reason, and graph-route
counters matched exactly, but attention and K/V diverged after the valid
four-row hidden table was exhausted. The attempt is retained as negative
evidence. It does not belong to the legal performance matrix and motivated the
pre-loop capacity check used by `raw-run3`.
