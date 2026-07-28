# Attempt 74 CANN 8.5.1 Formal Result

Date: 2026-07-29 (Asia/Shanghai)

## Scope

This result validates Cruise commit
`9e0e1209c6f2ce9e02479c33a1617a295cdc50d1`, installed from
`https://github.com/Li-changwu/Cruise.git` into a clean checkout. The package
was installed editable in the `vllm-hust-dev` Conda environment. The full
44-test suite, repository audit, CMake build, exact DataFlow symbol export,
source verifier, multi-epoch semantic verifier, and final result verifier all
passed.

The measured workload is Qwen2.5-7B-Instruct, TP=PP=1, one-token prompt,
static B=4, K=2, greedy sampling, and one Ascend 910B2 running CANN 8.5.1.
Four independent EngineCore/sidecar processes ran in blocked-ABBA order:

```text
old-1 -> new-1 -> new-2 -> old-2
```

Each block contains 15 measured epochs after its own initialization and
warmup, giving 30 old and 30 new observations.

## DataFlow Payload

The interposer reads `Tensor::GetSize()` from the actual
`DFlowSessionImpl::FeedDataFlowGraph` and `FetchDataFlowGraph` arguments. All
60 epoch windows contained exactly one Feed and one Fetch with the expected
ordered tensor-size sequence.

| Route | Feed tensors | Feed bytes | Fetch tensors | Fetch bytes | Total bytes |
|---|---:|---:|---:|---:|---:|
| old | 10 | 58,720,516 | 10 | 78,184,928 | 136,905,444 |
| new | 8 | 260 | 2 | 368 | 628 |

The observed reduction is 136,904,816 B per epoch, or 99.9995413%. It matches
the declared ABI ledger exactly, but the observed values are independently
derived from runtime tensor arguments rather than substituted constants.

## Runtime Activity

The tracer covers 16 `rtMemcpy`/`rtsMemcpy` variants, selected Mbuf/Buff APIs,
and the exact CANN 8.5 DataFlow C++ ABI symbols. Every process recorded 1,745
runtime memcpy records and 23 Mbuf diagnostic records during startup. None
occurred inside the 15 measured epochs of any block. Thus all 60 steady-state
epochs report runtime memcpy as `observed_zero`; they do not report a missing
measurement.

The installed CANN 8.5.1 application profiler passes a configuration that the
resident sidecar rejects during `GEInitialize`. Consequently, process-wide
profiler transfer bytes are `not_observed`. DataFlow tensor payload is not
claimed to equal PCIe, HCCS, DMA, or other physical-link traffic.

## Host Control Cost

| Metric | Old median | New median | Old/New |
|---|---:|---:|---:|
| Host-control wall time | 212.208 ms | 59.951 ms | 3.54x |
| Python CPU time | 2.368 ms | 1.045 ms | 2.27x |
| Native Feed-to-Fetch wall time | 129.674 ms | 59.186 ms | 2.19x |
| Native process CPU time | 154.240 ms | 1.163 ms | 132.62x |

Every epoch used four `add_request` calls, one EngineCore step, one post-step,
one socket send/receive pair, one DataFlow Feed/Fetch pair, and two device
model calls. The experiment imposes no speedup threshold; these values are
reported from the accepted 30+30 samples.

## Integrity

- Result SHA256:
  `163f103d9b2d16b180b2f840e94421b337df5589a26ac766d9727088467775d2`
- Transfer summary SHA256:
  `64ef1d915c1470d0d1a762bcce2eea16ec66f5d8f5f8fa4e8b8b7fbf27977519`
- Evidence manifest SHA256:
  `6cfe51a6570dff8b2f8cf7b89e0bfce2d032800eae70ee66ef36a650da682c0b`
- Compact external evidence: 107 hashed files, approximately 9.2 MiB.
- Raw transfer traces: approximately 828 KiB; filtered epoch rows:
  approximately 124 KiB.

All manifest entries passed `sha256sum -c`. The independent verifier passed
again after the 30.5 GB runtime scratch tree had been removed, proving that
the compact evidence retains every artifact needed to reconstruct the 60
transfer windows. The older 29 GB diagnostic scratch tree was also removed
through the marker-protected storage guard; its compact evidence remains.
