# M1 Continuous Admission Protocol

This gate exercises a real nontrivial-prefill trace across epoch boundaries:

1. A performs stock prefill, transfers to resident row 0, and executes K=2.
2. B arrives while A is Device-owned. A is isolated from the Host step while B
   alone performs stock prefill.
3. The next resident epoch keeps A Device-owned and imports only B into row 1.
4. B completes. C performs an isolated stock prefill, then reuses row 1 with a
   new generation while A remains Device-owned.
5. A completes through the unchanged steady ABI.

The stock and Cruise routes must match per-request tokens, terminal reasons,
stop reasons, and final scheduler accounting. The two isolated admission steps
must never schedule A on the Host. The B and C import checksums must match on
Host and Device, and C must reuse B's row with a strictly newer generation.

All generated models, weights, builds, caches, sockets, and logs remain in
marker-owned `/dev/shm` scratch and are deleted after bounded evidence is
finalized.
