# Project History

This document maps the source snapshots retained in `history/attempts/` to the
research questions they addressed. Attempt numbers are experiment identifiers,
not software releases.

## Stage 0: Device-Control Feasibility

The synthetic P0 established that one Host Feed/Fetch can enclose N recurrent
`RunFlowModel` calls in a Device UDF. The identical-artifact Host GE versus
Device UDF comparison crossed over at N=2 and reached 3.65x at N=32. The
real-Qwen layer-slice P0 then showed elementwise-identical recurrent attention
and KV state for N=1,2,4. The bounded controller added runtime graph selection,
EOS/max-step termination, and capacity rejection.

Sources: `experiments/synthetic-p0/`, `history/attempts/real-qwen-p0/`, and
`history/attempts/bounded-decode-controller/`.

## Stage 1: Full-Decoder Semantic Closure

Attempts 41-51 isolated sparse QK differences in the GE online-compiled path.
Attempts 52-65 progressively audited attention, BF16 materialization,
boundaries, transpose/layout choices, fusion, residual paths, and all linear
layers. Attempt 65 is the clean full-decoder packaging milestone used by later
DataFlow work.

Sources: `history/attempts/attempt41/` through
`history/attempts/g4a-full-attempt65-clean*/`.

## Stage 2: DataFlow and Device UDF Bring-Up

Attempts 66a and 66b moved from DataFlow smoke tests to BF16 Device UDF I/O,
all required operators, repeated decoder invocation, and persistent controller
state. These experiments defined the compiler/runtime constraints later used
by the resident epoch.

Sources: `history/attempts/g4b-attempt66*/`.

## Stage 3: Batched Full-Decoder Epochs

Attempts 67-70 established B=2 and then B=4 eager, AIR, native, and resident
epoch routes. They added active-row masking, full decoder logits, greedy
sampling, persistent Paged-KV, runtime graph metadata, recovery contracts,
storage guards, and the blocked-ABBA performance protocol.

The authoritative G4 B=4 performance run used one Feed/Fetch for the Device
route and K Host submissions for the Host route. Median paired speedups were
1.55x at K=2, 3.56x at K=4, and 5.36x at K=8. Full details and claim boundaries
are in `history/attempts/g4/G4-STATUS-20260724.md`.

Sources: `history/attempts/g4c-attempt67*/` through
`history/attempts/g4c-attempt70*/`.

## Stage 4: vLLM Integration

Attempt 71 introduced the scheduler contract and eligibility envelope. Attempt
72 connected one vLLM EngineCore step to the native sidecar. Attempt 73 proved
multi-epoch cohort evolution and safe row generation reuse with the trace
`[A] -> [A,B] -> [A,C]`.

Attempt 74, now at the repository root, removes dead Host-UDF Paged-KV and
diagnostic payloads. The boundary changes from 10 inputs/10 outputs and
136,905,444 declared bytes per epoch to 8 inputs/2 outputs and 628 declared
bytes. The decoder ABI and accepted Attempt 73 semantics remain fixed.

The formal CANN 8.5.1 blocked-ABBA run then observed the same old/new byte
counts on the actual DataFlow Feed/Fetch tensors for all 60 measured epochs.
Median Host-control wall time changed from 212.208 ms to 59.951 ms (3.54x).
No covered runtime memcpy or Mbuf event occurred inside any measured epoch;
all such records were startup-only. Physical-link bytes remain unobserved
because application `msprof` cannot initialize the resident sidecar on this
CANN release. The accepted result boundary is recorded in
`evidence/ATTEMPT74-CANN851-R5.md`.

Sources: `history/attempts/vllm-integration-attempt71-*` through
`history/attempts/vllm-integration-attempt73-*`, followed by the active root.

## Next Research Gate

The next gate is not another synthetic loop. It is to widen the current fixed
resident epoch into a controlled serving path without weakening correctness:
real prefill-to-decode transition, scheduler-compatible continuous admission,
cancellation/preemption semantics, and broader sampling. Each feature should
first preserve the one-epoch causal comparison and explicit fallback contract
before it is admitted into a full API-server evaluation.
