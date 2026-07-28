# G4c Attempt 69b: Fixed B=4 Batched Eager Semantics

Date frozen: 2026-07-24

## Question

Can one true static `B=4` decoder graph preserve four independent request
states, rather than invoking the accepted `B=1` graph four times at the epoch
controller level?

## Frozen ABI

- token: `[4, 1]`, INT64
- position: `[4]`, INT64
- sequence length: `[4, 1]`, INT32
- block table: `[4, 2]`, INT32
- slot mapping: `[4]`, INT32
- key/value Paged KV: `[28, 8, 128, 4, 128]`, BF16
- explicit QK tiling: `[72]`, UINT8
- active mask: `[4]`, INT32 with values in `{0, 1}`
- outputs: logits `[4, 1, 152064]`, updated key/value Paged KV, and next
  position `[4]`

Each request owns two disjoint physical blocks. The frozen block table is
`[[1,0],[3,2],[5,4],[7,6]]`. Logical capacity remains eight positions.

## Cases

1. Four active requests with starting positions zero, one, two, and three.
2. Two active requests alternating with two empty slots.
3. Finished, active, empty, and active slots in one batch.

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

This attempt cannot close AIR export, native GE, Device UDF execution,
performance, recovery, or vLLM-Ascend integration.
