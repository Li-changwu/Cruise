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

The M4b gate additionally requires one Feed/Fetch per resident epoch and a
direct Device IPC KV import: the Host snapshot checksum must be zero, the
Device checksum must be nonzero, and the import input must be the 43,200-byte
sidecar request plus IPC metadata. The 260-byte/368-byte steady epoch ABI is
unchanged. A 29,360,372-byte Host snapshot is accepted only by diagnostic
legacy validation and cannot pass this M4b gate. Heavy artifacts and build
products are allowed only in marker-owned `/dev/shm` scratch space.
