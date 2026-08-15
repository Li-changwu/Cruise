# Stage serial admission before concurrent Prefill

The first redesigned serving path uses Quantum-Boundary Serial Admission: the
Host may prepare request metadata and KV leases asynchronously, but the
Persistent Device Model Owner serializes bounded Prefill execution, KV transfer
acknowledgement, retirement, and cohort changes with Decode at an internal
Device Scheduling Quantum boundary. A Host-Visible Decode Epoch may be used as
a transition prototype, but cannot graduate as this stage's control
architecture. This stage makes no concurrency claim. The final architecture
must implement Concurrent Prefill/Decode on one NPU and prove real Device-task
overlap, bounded HBM, and preserved TTFT and TPOT; mere co-residence of Host and
sidecar processes does not satisfy that goal.
