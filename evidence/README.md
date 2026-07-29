# Evidence Policy

This directory is reserved for compact, reviewable evidence such as result
summaries, verifier output, source manifests, and hashes. Raw profiling trees,
AIR packages, weights, runtime logs, arrays, and scratch output are not stored
in Git.

The current accepted Attempt 74 measurement is summarized in
[`ATTEMPT74-CANN851-R5.md`](ATTEMPT74-CANN851-R5.md). Earlier measurements are
summarized in the frozen historical status documents. Original artifacts
remain outside this repository and must be matched through their recorded
SHA-256 values before use in a paper.

The current productization checkpoint is summarized in
[`M0-PRODUCT-READINESS-20260729.md`](M0-PRODUCT-READINESS-20260729.md). It
records both the completed contract/package checks and the still-open M0
real-model lifecycle condition.

The first M1 ownership/commit-state checkpoint is summarized in
[`M1-COMMIT-STATE-20260729.md`](M1-COMMIT-STATE-20260729.md). It records the
prepared/executing/committed boundary and explicitly excludes unqualified
prefill, API streaming, cancellation, and hardware fault claims.

The first real prefill ownership-transfer checkpoint is summarized in
[`M1-PREFILL-TRANSFER-20260729.md`](M1-PREFILL-TRANSFER-20260729.md). It records
the stock-vLLM token baseline, one-shot Paged-KV import checksum proof, first
Device-owned steady epoch, lifecycle cleanup, and the remaining M1 boundary.

The batched prefill checkpoint is summarized in
[`M1-BATCHED-PREFILL-20260729.md`](M1-BATCHED-PREFILL-20260729.md). It covers
simultaneous B=1-4 nontrivial prefills, mixed prompt/output lengths, multi-row
KV import checksums, completion-driven batch shrink, and exact stock-vLLM
differential results.

The continuous-admission checkpoint is summarized in
[`M1-CONTINUOUS-ADMISSION-20260729.md`](M1-CONTINUOUS-ADMISSION-20260729.md).
It covers isolated stock prefills while another request remains Device-owned,
selective Paged-KV import, completion, and generation-checked row reuse.
