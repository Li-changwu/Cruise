---
status: accepted
---

# Stage the Persistent Owner path

Persistent development proceeds in order: P0 proves weight-free control; P1
adds deterministic AICore compute and incremental outfeed; P2 proves admission,
cancellation, generation isolation, and retirement; P3 integrates the real
Decoder and a Device KV Lease of at least 384 tokens per target request; P4
builds Quantum-Boundary Serial Admission for the 128/256 streaming C4 service;
P5 qualifies Persistent Decode against Graph; and P6 adds and finally qualifies
Concurrent Prefill/Decode. A stage cannot begin until its predecessor's declared
evidence gate passes, so later model or concurrency work cannot hide a control,
transport, state-ownership, or performance failure.

P1 runs a deterministic AICore recurrence for 256 outputs per row at B=1 and
B=4, proving exact token, state, and monotonic commit sequences, pre-shutdown
visibility, and per-row Output Credit isolation in three independent NPU starts.
P2 covers cancellation before first commit, during execution, and after
completion; duplicate, reordered, stale-generation, and stale-owner events;
retirement-before-reuse; and at least 1000 in-flight cancellations satisfying
50 ms p95 and 100 ms p99. P1 through P4 initially advance each eligible row by
at most one token per Device Scheduling Quantum; the Device autonomously chains
the next quantum, and any later multi-step fusion requires profiler evidence.

P3 first proves B=1 short recurrence, then B=1 at the full 128-token prompt and
256-token output capacity, and finally B=4 with mixed remaining budgets, EOS,
output blocking, and cancellation. Token IDs, finish reasons, EOS positions,
Device positions, page cursors, KV ownership and checksums, and Committed Prefix
must match the Graph oracle with 100% Device Decode coverage in three
independent NPU starts; the Host cannot retain or advance per-token KV state.
P4 uses a closed-loop concurrency-four manifest that keeps four requests in
flight until 32 complete, with exactly 128 prompt tokens and 256 generated
tokens per request. Its primary performance case ignores EOS, while ordinary
EOS, short output, burst arrival, and overload remain semantic and regression
cases.
