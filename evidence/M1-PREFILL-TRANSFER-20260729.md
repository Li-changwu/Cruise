# M1 Prefill-to-Resident Ownership Transfer

Status: accepted as a narrow M1 correctness checkpoint on 2026-07-29. This is
not the full M1 exit gate or a production-readiness claim.

## Gate

This run answers one concrete question: can stock vLLM execute a nontrivial
prefill, transfer its Paged-KV state exactly once into Cruise, and then continue
greedy decoding in device-resident epochs without changing the generated token
sequence?

The tested request used prompt token IDs `[9707, 11, 358]` and requested four
output tokens. The baseline and Cruise ran in separate processes on physical
NPU 7 using the same content-addressed Qwen2.5-7B-Instruct model view.

## Environment

- Ascend 910B2, CANN 8.5.1, and the `vllm-hust-dev` conda environment;
- vLLM commit `ec4847981f2d4dda8343b3c4c90eeb173f8f8eb7`;
- vLLM-Ascend commit `e967f235ba66edb48a28a6d943aee9455fee70cf`;
- stock `vllm.v1.core.sched.scheduler.Scheduler` for the baseline;
- Cruise `ResidentEpochScheduler` with the stock `NPUWorker` and
  `NPUModelRunner` for the ownership-transfer run;
- TP=PP=1, synchronous scheduling, B=1 active row in the static B=4 graph,
  greedy sampling, block size 128, and epoch bound K=2.

The baseline source was not modified. Although the Cruise plugin package was
installed in the environment, no resident plan was attached to baseline
steps, so the wrapper delegated directly to the original `NPUWorker` method.

## Result

| Step | Baseline | Cruise route | Output | State evidence |
|---|---|---|---|---|
| 0 | stock prefill | stock prefill | `2776` | no resident plan |
| 1 | stock decode | import epoch, K=2 | `4460, 311` | Host-owned, import required, one Feed/Fetch |
| 2 | stock decode | steady epoch, K=1 | `1855` | Device-owned, no import, one Feed/Fetch |
| cleanup | no model call | no model call | none | both requests drained |

Both routes produced the exact sequence `[2776, 4460, 311, 1855]`.

The first resident epoch captured scheduler block 1 from all 28 stock K/V
layers, imported it into resident row 0, and compared independently computed
Adler-32 values before decoder execution:

- Host snapshot checksum: `3477654769`;
- Device resident-cache checksum: `3477654769`.

The import epoch declared 29,360,372 input bytes and 368 output bytes. The next
epoch retained the Attempt 74 steady ABI: 8 inputs, 2 outputs, 260 input bytes,
and 368 output bytes. The request generation remained 1 across the ownership
transition.

## Protocol Changes

- Python contract v3 identifies Host- versus Device-owned state and requests a
  one-shot KV import.
- Sidecar protocol v5 adds a 64-bit transfer ID; request/response sizes are
  136/352 bytes.
- The transfer file has an 80-byte versioned header, fixed 29,360,128-byte
  payload, import mask, row generations, and checksum.
- Native code validates the path, size, transfer ID, shape metadata,
  generations, and checksum, then fails in the `executing` state on an
  ambiguous import or mismatch.
- `NPUWorker.shutdown()` now closes the sidecar before worker teardown. The
  accepted run ended with no process on NPU 7 and no socket left behind.

## Verification

- dependency-light suite: 51 passed, 1 skipped because local Torch is absent;
- frozen server suite: 73 passed, 4 dependency warnings;
- CANN 8.5.1 native new/old bridges and servers: built successfully;
- Device UDF: built successfully;
- minimal ABI verifier: passed with the steady 260-byte/368-byte boundary;
- baseline, Cruise, and differential result gates: passed.

Raw compact results are retained under `evidence/m1-prefill-transfer/`:

| Artifact | SHA-256 |
|---|---|
| `baseline.json` | `dcab0abc56371bffb312014982109d87dd00fd55a144186bd3a3e46fc96ed2d5` |
| `cruise.json` | `4d442f652a9ed9a3ebd901973845f76449b366de04c2f9ef1de7238c7517242e` |
| `comparison.json` | `a08b857e194ccd124cd38bf6b534e018139916e7411f6cbb05ba2ee35357893a` |

## Storage Boundary

Runtime weights and GE external weights occupied about 15 GB each. Their peak
combined scratch use was 29 GB, entirely below a marker-owned `/dev/shm`
directory. Root filesystem utilization stayed at 56%. Generated weights,
caches, CANN logs, AIR, and build products are not retained in Git and were
deleted after compact evidence was preserved.

## Remaining M1 Work

This run covers one request, one three-token prompt, one import epoch, and one
steady epoch. It does not qualify continuous admission, mixed batches,
completion and row reuse after imported prefill, streaming, cancellation,
disconnects, general EOS boundaries, arbitrary prompt lengths, unsupported
feature fallback after mixed admission, or the 1,000-request M1 exit suite.
