# G4c Attempt 67a: Fixed B=2 Batched Eager Semantics

Date frozen: 2026-07-24

## Question

Can one true static `B=2` decoder graph preserve two independent request
states, rather than invoking the accepted `B=1` graph twice at the epoch
controller level?

## Frozen ABI

- token: `[2, 1]`, INT64
- position: `[2]`, INT64
- sequence length: `[2, 1]`, INT32
- block table: `[2, 2]`, INT32
- slot mapping: `[2]`, INT32
- key/value Paged KV: `[28, 4, 128, 4, 128]`, BF16
- explicit QK tiling: `[72]`, UINT8
- active mask: `[2]`, INT32 with values in `{0, 1}`
- outputs: logits `[2, 1, 152064]`, updated key/value Paged KV, and next
  position `[2]`

Each request owns two disjoint physical blocks. The frozen block table is
`[[1, 0], [3, 2]]`. Logical capacity remains eight positions.

## Cases

1. Both active, with starting positions zero and two.
2. One active request plus one empty inactive slot.
3. One finished inactive request plus one active request.

For every active request, the oracle is a separate execution of the accepted
`B=1` decoder implementation with the corresponding two-block cache slice.
Inactive requests are not executed by the oracle and must preserve their
position and all Paged-KV bytes.

## Pass Rules

- Every active-request logits tensor is finite and matches its independent
  `B=1` oracle at `rtol=5e-3, atol=5e-3`.
- Every active-request greedy token equals its independent oracle.
- The packed final key/value Paged KV matches the packed independent oracles
  at the same tolerance.
- Every byte outside active addressed slots remains elementwise exact.
- Empty and finished request cache slices remain elementwise exact.
- Position increments by one only for active requests.

## Claim Boundary

This attempt cannot close AIR export, Device UDF execution, independent EOS,
`B=4`, performance, recovery, or vLLM-Ascend integration.

