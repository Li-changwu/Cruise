# M1 Batched Prefill Ownership Transfer

Status: accepted as the third narrow M1 correctness checkpoint on 2026-07-29.
This is not the full M1 exit gate or a continuous-serving claim.

## Gate

This run asks whether Cruise can transfer several simultaneous stock-vLLM
prefills into the static B=4 resident graph while preserving independent
prompt positions, output budgets, Paged-KV blocks, row generations, and final
scheduler accounting. It covers batch sizes 1, 2, 3, and 4 in one EngineCore
lifecycle for each route.

The baseline and Cruise used the same four cohorts:

| Case | Prompt lengths | Output budgets | Cruise resident epochs |
|---|---|---|---|
| B=1 | `3` | `4` | import K=2, steady K=1 |
| B=2 | `2, 4` | `5, 3` | import K=2, steady K=2 |
| B=3 | `2, 3, 5` | `2, 3, 4` | import K=1, steady K=1, K=1 |
| B=4 | `2, 3, 4, 5` | `2, 3, 4, 4` | import K=1, steady K=1, K=1 |

Every request had at least two prompt tokens and two requested output tokens.
Each prompt/output pair stayed inside the current eight-position resident
capacity.

## Environment

- server alias `vllm-hust-lcw-16rc`;
- Ascend 910B2, physical NPU 7, and CANN 8.5.1;
- conda environment `vllm-hust-dev`;
- vLLM commit `ec4847981f2d4dda8343b3c4c90eeb173f8f8eb7`;
- vLLM-Ascend commit `e967f235ba66edb48a28a6d943aee9455fee70cf`;
- Qwen2.5-7B-Instruct, TP=PP=1, synchronous scheduling, greedy sampling;
- stock `NPUWorker` prefill and a static B=4 DataFlow decoder graph.

The stock parent process used the base CANN environment. The materialize,
barrier, and custom OPP roots were injected only into the Device UDF sidecar.

## Result

The baseline and Cruise matched exactly for all 10 requests:

| Case | Output token IDs |
|---|---|
| B=1 | `b1-r0: [2776, 4460, 311, 1855]` |
| B=2 | `b2-r0: [358, 2776, 4460, 311, 1855]`; `b2-r1: [264, 220, 16]` |
| B=3 | `b3-r0: [358, 2776]`; `b3-r1: [2776, 4460, 311]`; `b3-r2: [220, 16, 19, 1042]` |
| B=4 | `b4-r0: [358, 2776]`; `b4-r1: [2776, 4460, 311]`; `b4-r2: [264, 220, 16, 19]`; `b4-r3: [220, 16, 19, 1042]` |

Terminal finish reasons, stop reasons, request statuses, computed-token counts,
and output-token counts also matched. Each cohort used one simultaneous stock
prefill, one complete-batch KV import, and only Device-owned epochs afterward.

The B=3 active count changed `3 -> 2 -> 1`; B=4 changed `4 -> 3 -> 2`.
Surviving requests retained the same row and generation as earlier requests
completed:

- B=3 generations: `[4,5,6,0] -> [0,5,6,0] -> [0,0,6,0]`;
- B=4 generations: `[7,8,9,10] -> [0,8,9,10] -> [0,0,9,10]`.

This proves completion-driven batch shrink for an already admitted cohort. It
does not yet prove new admission or row reuse after a nontrivial prefill.

## Paged-KV and ABI Evidence

Every import used the fixed 29,360,372-byte input and one Feed/Fetch. Host and
Device Adler-32 checksums matched independently:

| Case | Host checksum | Device checksum |
|---|---:|---:|
| B=1 | `3477654769` | `3477654769` |
| B=2 | `4081853333` | `4081853333` |
| B=3 | `114816985` | `114816985` |
| B=4 | `116248357` | `116248357` |

All later epochs retained the 260-byte input and 368-byte output steady ABI,
with one Feed, one Fetch, and one sidecar request/response per epoch.

## Verification and Storage

- full frozen server suite: 78 passed, 4 dependency warnings;
- CANN native bridge, server, and AIR relocation targets: built successfully;
- baseline, Cruise, and differential gates: passed;
- final NPU state: no process on physical NPU 7;
- scratch peak at finalization: 30,598,217,728 bytes (28.50 GiB);
- persistent evidence: 2.0 MiB; root filesystem remained at 56%;
- marker-owned runtime scratch was deleted after evidence finalization.

Compact results are retained under `evidence/m1-batched-prefill/`:

| Artifact | SHA-256 |
|---|---|
| `baseline.json` | `efa06469b506045283576fe527394dfea8266af9459828571b6630d6f25dbd74` |
| `cruise.json` | `2652a62f37e062bec745458e4ef5e264aea0067c5f0772c43d9ae943049d69dc` |
| `comparison.json` | `4b2fd13bd70e8f68b53533ed3befff891b05d63e0102bea1eefeb20cb2755301` |

## Remaining M1 Work

The next gate is continuous admission at epoch boundaries: admit a new
nontrivial prompt after another request completes, reuse the released row with
a new generation, and preserve both imported and already Device-owned KV state
in one mixed epoch. Unsupported-request routing, OpenAI streaming and
non-streaming order, EOS, disconnect/cancellation, and the 1,000-request exit
suite remain open.
