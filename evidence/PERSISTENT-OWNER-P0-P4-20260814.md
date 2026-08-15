# Persistent Device Model Owner P0-P4 checkpoint

Status: P0-P4 are Candidate Hardware Validated on physical NPU 0. This is a
research-track checkpoint, not M2 lifecycle/soak, M3 operator observability,
M4 performance qualification, or Stable v1.0 evidence.

## Gate results

| Gate | Authoritative run | Accepted result |
|---|---|---|
| P0 | `persistent-control-p0-profile-20260808-r14` | Three mechanism starts plus one profiled start passed persistent control, placement, quiescence, release, and integrity checks. |
| P1 | `persistent-recurrence-p1-profile-20260809-r4` | Three mechanism starts plus one profiled start produced exact recurrence and 752 attributed AICore calls; the authoritative result is the reanalysis that retained the rejected original. |
| P2 | `persistent-cancel-p2-profile-20260809-r5` | Three mechanism starts plus one profiled start passed generation isolation, retirement, 1,000 cancellations, latency, placement, release, and integrity checks. |
| P3 | `persistent-decoder-p3-three-start-gate-20260814T114811Z` | Three cold B=4 Owner starts matched the cold Graph oracle exactly with 100% Device Decode coverage. |
| P4 | `persistent-admission-p4-gate-20260814T125100Z` | The 32-request C4 primary and lifecycle regression passed; the primary emitted 8,192 commits with exactly 3,064 AICore calls. |

P0-P2 source hashes in the accepted evidence match the corresponding source in
the checkpoint branch before commit. The accepted P3/P4 runs predate later P5
source evolution. Their historical identities are preserved below; they must
not be attributed to the newer committed source without a replay.

## Source identity

| Scope | Host SHA-256 | Controller SHA-256 | Relationship to checkpoint source |
|---|---|---|---|
| P0 | `89399661186bbee16f726e914cc068a9025f16df26e5198a4976d6ec7a6c97c6` | `da5dc7f21ef5a92719c1d26a551bffd58ef8eae301ce1f260091c91b7c27517a` | Exact |
| P1 | `e676f35b932a3309fd9f25e91d7bfab3d621ec0f560980ed8f80d6f2778c24f5` | `9f33ad760c4d9ba44d8983be507bd63457993bb96327e494d41433043cbbd5e5` | Exact |
| P2 | `a0d48efbf6abb54be356a2c147a5b74b1570f05591d2db4511ad5ee364b6614c` | `19ea59600ef190e07b9569cdff6ff212ad535768b223527c3d0a50a21fcf55f2` | Exact |
| P3 accepted run | `a478115c4f06605b2f2ad31b0738ca0bc49386c999d900b6ee141bf281a0f705` | `8a52f9e429d5a08fa09223c3319cd06174eef143a6085645a81404ae741c859d` | Historical; replay required |
| P4 accepted run | `dba9f2ed81fa18db6e862d4773dad70c2064981a9cdde226b8e3c0b1e6142bc1` | `2ffac90c944a7d5c49714f63e9584f0a5026647f7d09e9d68ece941a5a47b616` | Historical; replay required |

P3 used AIR SHA-256
`53e7c8407ac638af4c8b50865e8cc1c5ea428aedfa399b56c9bf014b792cc60b`
and external-weight bundle identity
`f93a2828d8b3ded10acc3831d114abe8b57738ad3adfc357a0f2bb047939866b`.

## Compact evidence integrity

| File | SHA-256 |
|---|---|
| `p0-result.json` | `b47f47dbe5f18738f90923372a232a3576203662361ab109334147d0e2e03457` |
| `p1-result.json` | `d03752b7e6fd91e3dcbb516ca7cba9842c2abdba5c0dbef5b0d24f672bd8bb63` |
| `p2-result.json` | `2347c46674767210d5d9476b66b056f825e02429b8d2d4833d5ff0aba0101e28` |
| `p3-three-start.json` | `d5d8a82965ac57f619dbc81baf3fc582eb7496b4848f9cb1220d9dffde6174c8` |
| `p4-gate.json` | `f9bbddde9982a0d8da5a9b4bcefced735185497ca8b68f98a2f1d11d80c349cc` |

The raw logs, model, AIR package, external weights, mutable GE views, compiler
products, and profiler trees remain outside Git under the documented storage
lifecycle. The retained JSON is sufficient to review gate decisions but not to
relabel a different source revision as the measured candidate.
