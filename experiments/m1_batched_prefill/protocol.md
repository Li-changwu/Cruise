# M1 Batched Prefill Differential Protocol

This gate runs four cohorts with batch sizes 1, 2, 3, and 4. Every request
has a nontrivial prompt. The B=2-4 cohorts use mixed prompt and output lengths,
and all prompt/output combinations stay inside the current eight-position
resident support boundary.

For each cohort, the baseline and Cruise run the same requests in separate
processes. Cruise must execute the simultaneous stock prefill first, import all
active scheduler Paged-KV blocks in one generation-checked transfer, and use
only device-owned resident epochs afterward. Per-request token IDs, terminal
finish reasons, stop reasons, and final scheduler accounting must match the
stock baseline exactly.

The gate additionally requires one Feed/Fetch per resident epoch, matching
Host and device Adler-32 import checksums, a 29,360,372-byte import input, and
the 260-byte/368-byte steady epoch ABI. Heavy artifacts and build products are
allowed only in marker-owned `/dev/shm` scratch space.
