# P4 Quantum-Boundary Serial Admission protocol

P4 turns the P3 Persistent Device Model Owner into a bounded service driver.
It evolves the single P3 controller source with Device-owned staged cohort
activation and reuses the validated Decoder AIR. P4 adds Host request admission and asynchronous output drain, but it does not fork the Owner or move
authoritative Decode progression back to the Host.

## Primary gate

The formal `primary-c4` workload has 32 requests. Every request has exactly 128
prompt tokens and a 256-token output budget, ignores EOS, and streams every
Committed Prefix token as a separate output. The Host initially admits four
requests and submits one replacement only after observing a retirement. It may
send admission and shutdown events, but it sends no token-step, Decode-position,
page-cursor, or KV-progression event.

The gate requires:

- exactly 32 completed requests and 8,192 committed output tokens;
- at most four requests pending or active in the closed loop;
- exactly 33 Host feeds: 32 admissions and one shutdown;
- exactly one streamed output message per committed token;
- 3,064 AICore Decoder calls, or eight four-row cohorts times the independent
  383-call cold Graph oracle recurrence;
- every request's tokens, finish reason, final position, page cursor, and KV
  checksum exactly match the pinned cold Graph oracle;
- all rows retire before reuse and the Owner remains alive until the final
  shutdown event.

The primary workload records admission, first-token, every-token, and retirement
timestamps. These timings establish service observability only. They are not a P5 Host CPU, TPOT, throughput, or TTFT qualification result.

## Regression gate

One model-load-scoped `regression` run covers eight short-output requests, one
ordinary primary-EOS request, two C4 bursts separated by 100 ms, an eight-way
overload against four rows followed by successful retry, and cancellation after
one committed token. Exactly four overload admissions must be rejected with
`NO_CAPACITY`; every rejected request must later be admitted and completed.
Credit events are intentionally asynchronous, so this regression accepts 110 through 131 AICore calls depending on whether already-credited rows advance
before the remaining credit events arrive. The primary staged cohort gate keeps
its exact 3,064-call requirement.

P3 lifecycle evidence remains the inherited baseline. The runner records the
evolved controller hash, and the P4 regression reruns cancellation, Output
Credit, capacity rejection, retirement, and row reuse so a P4 result cannot
silently rely on unvalidated lifecycle behavior.

## Claim boundary

P4 proves Quantum-Boundary Serial Admission, incremental output drain, bounded
capacity, and service semantics. It does not claim that Prefill and Decode run concurrently. P5 performs same-round Graph performance qualification; P6 is the
first phase allowed to claim real Prefill/Decode overlap.

## Current validated state

P4 is Candidate Hardware Validated on physical NPU 0. The formal `primary-c4`
run `persistent-admission-p4-primary-c4-20260814T123734Z` completed 32 requests,
8,192 single-token output messages, 33 Host feeds, and exactly 3,064 AICore
Decoder calls. All four rows were reused eight times, and every token, finish
reason, final position, page cursor, and KV checksum matched the pinned cold
Graph oracle.

The model-load-scoped regression run
`persistent-admission-p4-regression-20260814T122710Z` completed all 26 requests
and 194 commits with four expected `NO_CAPACITY` rejections, four Output Credit
acknowledgements, natural EOS, burst admission, successful overload retry, and
cancel-after-commit. Both successful runs used byte-identical source identities
and retired their mutable weight views and tmpfs scratch.

The authoritative aggregate is
`persistent-admission-p4-gate-20260814T125100Z`. P4 timings remain
observability-only until P5 completes three independent same-round Graph versus
Owner starts with whole-process-tree Host CPU attribution.
