# M1 Exit Differential Protocol

The gate generates 100 cohorts at each batch size B=1,2,3,4, for exactly
1,000 deterministic requests. Prompt lengths cover 2-5 tokens. Output budgets
cover every value that fits the current eight-position resident capacity.

Each request is assigned one deterministic semantic class: ordinary greedy
decode, unsupported `min_tokens=1` Host routing, EOS on the known second output
token, cancellation immediately after stock prefill, or cancellation after
three emitted tokens and Device ownership. Cancellation is applied at the same
per-request token boundary in the baseline and Cruise runs.

The baseline and Cruise must match every request's token IDs, terminal reason,
stop reason, and final scheduler accounting. Cruise additionally requires:

- no Host execution of Device-owned state;
- no Device epoch containing an unsupported request;
- at least one short Device epoch terminated by EOS;
- at least one cancellation after Device ownership;
- generation-checked row reuse without lease aliasing;
- one Feed and one Fetch per Device epoch;
- matching Host/Device checksums for every Paged-KV import.

Models, builds, external weights, caches, sockets, and logs remain in
marker-owned `/dev/shm` scratch. Only bounded JSON results and hashes persist.
