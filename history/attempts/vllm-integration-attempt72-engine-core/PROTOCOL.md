# Attempt 72: EngineCore Resident-Epoch Execution Backend

Date frozen: 2026-07-27

## Gate objective

Prove that the accepted fixed B=4 DataFlow resident-epoch kernel is callable
as an execution backend from the real vLLM v1 engine stack:

```text
EngineCore
  -> ResidentEpochScheduler
  -> stock UniProcExecutor
  -> WorkerWrapperBase
  -> ResidentEpochWorker
  -> DataFlow sidecar
  -> Device UDF
```

Attempt 71 r9 already proved the real SchedulerOutput contract on physical NPU
7. Attempt 72 changes only the engine/executor entry path. The Qwen2.5-7B
decoder AIR, B=4 Device UDF, greedy/EOS semantics, Paged-KV layout, frozen
baseline, tiling, and external-weight materialization remain unchanged.

## Frozen support boundary

- Qwen2.5-7B-Instruct, TP=PP=1, synchronous scheduling;
- one-token prompts that finish in one resident epoch;
- proven logical batch sizes B=1, B=2, and B=4, padded to one B=4 graph;
- a shared max-step budget K in 1, 2, 4, or 8 for active requests;
- greedy sampling with primary EOS and one independent-EOS B=4 case;
- no preemption, request insertion, continuous batching, cache connector, LoRA,
  multimodal input, structured output, logprobs, penalties, or speculation.

## Controlled case matrix

The runner must use `EngineCore.add_request()`, `EngineCore.step()`, and
`EngineCore.post_step()` for all 13 cases:

- B=1/2/4 crossed with K=1/2/4/8;
- B=4, K=4 with rows 0 and 2 terminating at the first generated token while
  rows 1 and 3 run to the length cap.

Each request starts from the same frozen input token and is compared against
the accepted Attempt 69e-r5 native baseline. No deterministic test backend is
permitted in the native step.

## Pass rules

1. vLLM resolves exactly `ResidentEpochScheduler`, stock `UniProcExecutor`,
   and `ResidentEpochWorker`.
2. World-size-1 Gloo and model-parallel state are initialized while EngineCore
   is live and destroyed by `EngineCore.shutdown()`.
3. All 13 cases return the frozen token sequences and correct length/EOS finish
   reasons through EngineCore output objects.
4. Scheduler computed-token and output-token accounting equals every request's
   actual Device executed-step count.
5. Every resident epoch reports exactly one DataFlow Feed and one Fetch; model
   calls equal the maximum executed steps in that epoch.
6. A second control-only EngineCore step drains finished requests, emits no
   token output, and performs no model execution.
7. `verify_engine_core_result.py` independently rejects missing cases, class
   substitutions, failed cleanup, missing artifacts, or a leaked distributed
   environment.
8. The pinned vLLM and vLLM-Ascend commits, native binaries, source tree,
   runtime AIR, 342 runtime weights, 379 GraphPp external-weight files, result,
   and evidence directory all receive integrity records.

## Storage and device invariants

- The guarded launcher is the only allowed entry point.
- Builds, relocated AIR, weights, caches, logs, sockets, and temporary files
  live below marker-protected `/dev/shm/a72r1`.
- Root reserve is at least 100 GiB, `/dev/shm` reserve at least 128 GiB,
  persistent project use at most 24 GiB, evidence at most 512 MiB, and scratch
  at most 64 GiB.
- Logs are bounded, existing evidence is never overwritten, and unexpected
  persistent growth above 64 MiB terminates the active command.
- NPU 7 must have no process and at most 5 percent HBM use for three consecutive
  samples before graph loading and after EngineCore shutdown.
- Scratch is removed only after the result verifier, NPU-ready check, evidence
  hashing, and storage finalization all pass.

## Claim boundary

Passing closes the fixed-support-envelope objective: the resident epoch is a
real vLLM Scheduler/EngineCore execution backend on this 910B2/CANN 9.0.0
server. It does not prove API-server integration, Host fallback routing,
multi-epoch request residency, dynamic request insertion, shared vLLM/DataFlow
KV allocation, or end-to-end serving performance.
