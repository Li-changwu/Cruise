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
