# M1 Continuous Admission and Row Reuse

Status: accepted as the fourth narrow M1 correctness checkpoint on 2026-07-29.
This is not the full M1 exit gate or a general continuous-serving claim.

## Gate

This run asks whether Cruise can admit a new nontrivial stock-vLLM prefill
while an existing request remains Device-owned, import only the new Paged-KV
row into the next resident epoch, and safely reuse a completed request's row
with a new generation.

The fixed trace was:

1. A performs stock prefill, imports into row 0/generation 1, and executes K=2.
2. B arrives after A has produced three tokens. The Host step schedules only B;
   A remains paused and Device-owned.
3. A and B execute a mixed resident epoch. Only B imports Paged-KV into row
   1/generation 2, and B completes.
4. C arrives, performs an isolated stock prefill, and joins A in a mixed epoch.
   C reuses row 1 with generation 3, then completes.
5. A finishes in an unchanged Device-owned steady epoch.

All prompts had at least two tokens. Output budgets were seven tokens for A
and two tokens each for B and C, within the current eight-position capacity.

## Environment

- server alias `vllm-hust-lcw-16rc`;
- Ascend 910B2, physical NPU 7, and CANN 8.5.1;
- conda environment `vllm-hust-dev`;
- vLLM commit `ec4847981f2d4dda8343b3c4c90eeb173f8f8eb7`;
- vLLM-Ascend commit `e967f235ba66edb48a28a6d943aee9455fee70cf`;
- Qwen2.5-7B-Instruct, TP=PP=1, synchronous scheduling, greedy sampling;
- stock `NPUWorker` prefill and a static B=4 DataFlow decoder graph.

## Result

The stock baseline and Cruise matched exactly:

| Request | Output token IDs | Final accounting |
|---|---|---|
| A | `[358, 2776, 4460, 311, 1855, 264, 2025]` | computed 8, output 7 |
| B | `[2776, 4460]` | computed 4, output 2 |
| C | `[264, 220]` | computed 5, output 2 |

All three requests finished as `FINISHED_LENGTH_CAPPED`, with finish reason 1
and no stop reason. The scheduler drained all requests and the cleanup step did
not execute the model.

The Cruise route used exactly three Host model steps, with request sets `{A}`,
`{B}`, and `{C}`. The four Device epochs used `{A}`, `{A,B}`, `{A,C}`, and
`{A}`, with epoch lengths 2, 1, 1, and 2. A never appeared in either isolated
Host prefill after it became Device-owned.

## Paged-KV and Lease Evidence

| Imported request | Row/generation | Host checksum | Device checksum |
|---|---|---:|---:|
| A | 0/1 | `1106791497` | `1106791497` |
| B | 1/2 | `3477654769` | `3477654769` |
| C | 1/3 | `2438977342` | `2438977342` |

A retained row 0/generation 1 throughout. C reused B's released row only after
the lease generation advanced from 2 to 3. Every Device epoch used one Feed
and one Fetch. The final A-only epoch performed no KV import and retained the
260-byte input and 368-byte output steady ABI.

The scheduler now fail-stops if a Host step without a resident plan contains a
Device-owned request. This prevents stale Host KV from advancing a request,
but it is not yet the supported Host fallback required for ineligible traffic.

## Verification and Storage

- local dependency-light suite: 58 passed;
- full frozen `vllm-hust-dev` suite: 83 passed, 4 dependency warnings;
- CANN native bridge, server, and AIR relocation targets: built successfully;
- baseline, Cruise, and differential gates: passed;
- scratch peak at finalization: 30,598,225,920 bytes (28.50 GiB), entirely in
  marker-owned `/dev/shm`;
- persistent server evidence: 1.80 MiB; root filesystem remained at 56%;
- runtime scratch was deleted after evidence finalization.

Compact results are retained under `evidence/m1-continuous-admission/`:

| Artifact | SHA-256 |
|---|---|
| `baseline.json` | `4209c40cd2fe30589a471936b02b069d79b0704b12da4efaa8e88985e48029c0` |
| `cruise.json` | `491cf829609fcc07a2b17d1b8e053af8bcf9cc17ffde3672677fd8e1f09973d6` |
| `comparison.json` | `621728331739e07eee96307506bda407f7b26d44c977cc2fbb9bafd963864245` |

## Remaining M1 Work

This checkpoint covers one eligible waiting request per admission boundary. It
does not cover multiple simultaneous arrivals, requests outside the resident
eligibility envelope, API streaming/non-streaming order, EOS, disconnect,
cancellation, or the 1,000-request differential exit suite. The next ordered
gate is stock-equivalent routing for unsupported requests before any device
ownership transfer.
