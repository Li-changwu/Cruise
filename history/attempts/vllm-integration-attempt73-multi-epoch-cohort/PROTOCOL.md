# Attempt 73: Multi-Epoch Residency, Cohort Update, and Warmup

Date frozen: 2026-07-27

## Gate objective

Prove that one real vLLM `EngineCore` and one persistent DataFlow Device UDF
can execute multiple resident epochs while preserving request-local decoder and
Paged-KV state across Host scheduling boundaries. New requests may enter and
replace completed rows only at epoch boundaries.

This gate extends the accepted Attempt 72 execution path. It does not repeat
the already closed question of whether Scheduler and EngineCore can call the
resident backend.

## Frozen trace

The runner uses one EngineCore, one sidecar process, and one Device UDF:

1. Epoch 1: add A with `max_tokens=6`; execute A at row 0, generation 1.
2. Epoch 2: add B with `max_tokens=2`; continue A at row 0/generation 1 and
   execute B at row 1/generation 2.
3. Epoch 3: add C with `max_tokens=2`; continue A at row 0/generation 1 and
   reuse row 1 for C with generation 3.
4. Cleanup: drain the finished scheduler without model execution.

Every service epoch has K=2 and uses exactly one DataFlow Feed and one Fetch.
The frozen greedy token sequence is:

```text
17728, 374, 264, 4185, 19734, 24844
```

A must produce all six tokens across the three epochs. B and C must each
produce the first two tokens.

## Resident-state contract

- Scheduler assigns a stable physical row to every continuing request.
- Every newly admitted request receives a monotonically increasing generation.
- The Device UDF accepts a continuing row only when generation, token,
  position, and sequence length match its committed resident metadata.
- A changed generation is accepted only from position zero. Before execution,
  both Paged-KV blocks owned by that row are cleared on device.
- Decoder outputs become authoritative resident K/V state only after the full
  epoch succeeds. Failed execution cannot commit partial state.
- The device returns the generation vector for independent Host validation.

## Explicit warmup

`ResidentEpochWorker.compile_or_warm_up_model()` must execute one synthetic
B=1/K=1 decoder step during EngineCore initialization through the same sidecar,
FlowGraph, Device UDF, and `RunFlowModel` path used by service epochs. Warmup
uses reserved generation `INT32_MAX`; the first real request receives
generation 1 and clears the warmup row.

The first service epoch must complete in less than 10 seconds. The approximately
552-second lazy GraphPp load observed in Attempt 72 must therefore be charged
to EngineCore initialization rather than the first service request.

## Pass rules

1. Warmup reports status zero, one model call, one Feed, and one Fetch.
2. The first service epoch is below 10 seconds.
3. The exact cohort sequence is `[A] -> [A,B] -> [A,C]`.
4. A remains row 0/generation 1; row 1 changes generation 2 to generation 3.
5. Epoch input positions are A: 0, 2, 4 and B/C: 0.
6. All generated tokens equal the frozen baseline.
7. Every service epoch reports two model calls and one Feed/Fetch pair.
8. Scheduler accounting ends at A=6, B=2, C=2 output/computed tokens.
9. Cleanup emits no output and performs no model execution.
10. The independent verifier, unit tests, integrity checks, NPU idle checks,
    storage finalization, and scratch cleanup all pass.

## Storage and device invariants

- The only allowed entry point is:

  ```bash
  /root/ascend-control-g4-20260723/storage-control/run_guarded_attempt.sh \
    /root/ascend-control-g4-20260723/attempt73-src
  ```

- Build products, relocated AIR, runtime weights, GraphPp external weights,
  caches, logs, sockets, and temporary files live below `/dev/shm/a73r2`.
- Root reserve is at least 100 GiB, `/dev/shm` reserve at least 128 GiB,
  persistent G4 use at most 24 GiB, evidence at most 512 MiB, and per-attempt
  scratch at most 64 GiB.
- Existing evidence is never overwritten. Successful scratch is deleted only
  after result verification, evidence hashing, final NPU idle, and storage
  finalization pass.

## Claim boundary

Passing proves multi-epoch device residency and epoch-boundary cohort updates
for the frozen Qwen2.5-7B, TP=PP=1, one-token-prompt, greedy, B<=4, K=2 trace on
one 910B2/CANN 9.0.0 server. It does not prove arbitrary prefill, preemption,
continuous batching, API-server integration, general sampling, cross-platform
generality, or end-to-end production serving performance.
