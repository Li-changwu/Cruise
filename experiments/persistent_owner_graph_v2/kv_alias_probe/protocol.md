# V2 KV alias probe

This weight-free probe asks one narrow question before a full model graph is
changed: can the public `ScatterPaKvCache` export update B4/K384 Paged-KV inputs
as direct `RefData`, without a cache-path `TensorMove`?

The probe uses the stock PA-NZ cache layout, four rows, three 128-token blocks
per row, four KV heads, and 128 values per head. Eager execution must update
exactly four declared slots in both caches. The exported graph must contain one
`ScatterPaKvCache`; its key-cache and value-cache inputs must both come directly
from `RefData` nodes.

This is only an export and structure gate. A pass does not prove ordinary Graph
or public GraphPp execution, zero Device copies, Prefill/Decode shared state, a
full Decoder, or performance. Those claims require later gates and separate
evidence. The diagnostic run retains the AIR SHA-256 and graph text, not the AIR
binary; a Graph/GraphPp pair must export and execute its temporary artifact in
the same bounded run or promote it through the asset lifecycle.

The V2 runner intentionally does not use the historical 5% idle-HBM line. It
requires no visible Device process, three stable preflight samples, a 65% safety
ceiling that catches retained model allocations, and post-run HBM recovery to
the sampled starting baseline with a 64 MiB default driver-variance allowance.

```bash
bash experiments/persistent_owner_graph_v2/kv_alias_probe/run_export_on_910b.sh
```

After the structure gate passes, the bounded pair runner exports a fresh
temporary AIR and executes it once through ordinary `ge::Session::RunGraph` and
once through the public DataFlow `GraphPp` path. Both modes must return the two
KV caches bitwise equal to an independently generated oracle and equal to each
other. The AIR and full cache binaries are deleted with scratch; only their
hashes, exactness summaries, component times, source identity, and bounded logs
are retained.

```bash
bash experiments/persistent_owner_graph_v2/kv_alias_probe/run_graphpp_pair_on_910b.sh
```

Passing this pair proves only single-layer B4/K384 KV update execution. It does
not yet prove a Prefill/Decode shared KV lease, absence of Device copies, the
full Decoder graph, FIA GraphPp compatibility, or P5 performance.
