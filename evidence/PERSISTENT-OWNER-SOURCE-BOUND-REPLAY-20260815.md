# Persistent Owner committed-source replay

Status: P3 and P4 are source-bound to commit
`950df7ca2d0667b7c04e151a75c65d62b2ee1fd3` on their accepted legacy compute
plane. The P5 FIA compute plane remains blocked. This checkpoint is not P5
performance qualification and does not open P6, M2, or M3.

## P3 replay

Three independent physical-NPU-0 cold starts used the accepted AIR SHA-256
`53e7c8407ac638af4c8b50865e8cc1c5ea428aedfa399b56c9bf014b792cc60b`
and external-weight bundle identity
`f93a2828d8b3ded10acc3831d114abe8b57738ad3adfc357a0f2bb047939866b`.
Every start produced the same summary SHA-256
`c8633b1c43b2cccfcd3da9f08e3cff9e3aa9bf45931bd4d73f2f91a8073d38d2`
and comparison SHA-256
`fa03cb8a0a0161c54a8fbfd13a3d07851cd933f89fd7a57ee7a54639cafb0894`.
Each run completed 4 requests, 5 Feed calls, 383 AICore calls, 344 commits,
and 4 retirements with 100% Device Decode coverage and no oracle mismatch.

## P4 replay

The source-bound primary completed 32 requests, 8,192 commits, 33 Feed calls,
and exactly 3,064 AICore calls with 100% coverage and no mismatch. The
source-bound regression completed 26 requests and 194 commits, including four
expected capacity rejections and retries, four credit acknowledgements, EOS,
burst admission, cancellation, retirement, and row reuse. The primary and
regression source-identity records were byte-identical. The aggregate passed.

## P5 boundary

A separate current-source replay used the FIA AIR SHA-256
`423123e4eabc0b0784a98386960b4af1a0db465e6ac2ab367ddb80461383e48d`.
It failed before the first admission Feed completed: GraphPp could not restore
the `FusedInferAttentionScore` kernel and reported
`BinaryGetFunctionByEntry failed, funcEntry=0` with runtime status `107000`.
The run emitted zero Feed calls, AICore calls, and commits. This reproduces the
known supported-compute-plane blocker; it is not a P3 legacy-path regression.

All five successful P3/P4 runs and the bounded FIA failure retired their
mutable external-weight views and tmpfs scratch. The final NPU state had no
visible process and reported 5% HBM usage. Raw CANN logs and compiler products
remain outside Git under the storage lifecycle.
