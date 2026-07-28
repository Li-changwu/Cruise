# vLLM Scheduler to Resident Epoch Contract

Date frozen: 2026-07-27

## Gate objective

Make the accepted fixed-batch G4 kernel callable from the real vLLM v1
scheduler without claiming request insertion, preemption, continuous batching,
or a shared vLLM/DataFlow KV allocator.

## Scheduler-owned state

The scheduler retains admission, fairness, request lifetime, stop reporting,
and KV block ownership. For an admitted epoch it emits, per request:

- current token, position, sequence length, EOS token;
- scheduler KV block IDs and a deterministic mapping to the static graph rows;
- the fixed B=4 graph (unused rows are inactive) and a global `K` in
  `{1, 2, 4, 8}`.

`num_scheduled_tokens` remains one because only the first input token is known
on Host. In the current G4 graph, logical capacity is 8 while the vLLM block
size is 128, so the physical block allocated for that first token already
covers every position the epoch can write. No extra block is allocated. A
future graph that can cross a block boundary must reserve `K-1` lookahead
positions before it may use this contract.

## Device-owned state

The backend executes K sequential decoder steps, greedy sampling, EOS handling,
and Paged-KV updates. It returns all generated tokens and the actual number of
computed steps for every request. The scheduler advances
`num_computed_tokens` by `actual_steps - 1` before normal vLLM output handling.

## Attempt 71 support boundary

- Qwen2.5-7B-Instruct native graph, TP=PP=1;
- one to four requests, padded to the single static B=4 graph;
- one-token prompts whose requests finish inside one epoch;
- equal max-step budget across active requests;
- greedy sampling with primary EOS only;
- synchronous scheduler and no cache connector, LoRA, multimodal input,
  structured output, logprobs, penalties, or speculative decoding.

Unsupported requests remain on the ordinary Host path. A request is never
moved to the Device path after Host has created its KV state. This prevents
silently mixing incompatible vLLM and G4 cache layouts.

The dedicated `ResidentEpochWorker` intentionally does not create a PyTorch
NPUModelRunner or allocate a second physical KV cache. In a resident-only
deployment, requests outside the support envelope must be rejected before
execution. Host fallback therefore requires a separate ordinary worker route;
it is not implemented by loading both models into one process.

## Completion evidence required

Attempt 71 is not complete until all of the following pass:

1. contract and accounting unit tests against the pinned vLLM tree;
2. real SchedulerOutput drives B=1/B=2/B=4 native execution;
3. returned tokens, finish reason, and scheduler token accounting match a Host
   baseline for K=1/2/4/8 and controlled EOS;
4. unsupported semantics perform zero Device model calls and use Host output;
5. one scheduler execute call produces one DataFlow Feed and one Fetch;
6. all artifacts pass the existing storage guard and evidence hashing policy.

Attempt 71 completed these requirements in r9. Attempt 72 raises the gate to
the real `EngineCore` and stock `UniProcExecutor`: configuration must resolve
`ResidentEpochScheduler` and `ResidentEpochWorker`, `EngineCore.add_request()`
must admit the requests, and one `EngineCore.step()` must return the complete
resident epoch for the same B=1/B=2/B=4, K=1/2/4/8, and independent-EOS suite.
