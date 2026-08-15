# Controller-Aware Graph V2 Device-state handle probe

The authoritative target-hardware run is
`persistent-owner-v2-device-kv-update-20260815T191338Z`, bound to clean source
commit `7055eb4f1096964ffcc931c5d0763957255ac25d`. Evidence integrity and the
workspace storage audit passed.

## Result

`V2-DEVICE-KV-UPDATE-LIFETIME` passed on physical NPU 0 of the pinned Ascend
910B2/CANN 9 stack.

- The exported AIR contained one `DeviceKvSlotUpdate`, two ordinary `Data`
  inputs, no external `RefData`, no `TensorMove`, and only a nine-element report
  output. Graph SHA-256 was
  `e8bcb0b294ba81f89e7c606a5b4be8230689004f155b9cfade69c7cebd2596f5`.
- Ordinary Graph loaded and executed the custom AICore kernel exactly on an
  8 KiB synthetic buffer. It returned no full-buffer output.
- FunctionPp allocated one Device buffer and retained one `FlowMsg`. Two
  `RunFlowModel` calls used the same message identity and data address. The first
  call changed slot 2,048 from BF16 zero to 1.0; the second observed 1.0 and
  changed it to 2.0.
- Both complete-buffer scans were exact. The first post-update Adler-32
  `3478782396` equalled the second pre-update checksum. Host cache input and
  output bytes were both zero; only 32 metadata-input bytes and 352 summary-
  output bytes crossed the Host-facing DataFlow boundary.
- The public `FlowMsg` and `MetaRunContext` header hashes were fixed. No raw
  Device address was encoded in a tensor, and no private `ModelPp` was used.

The isolated custom package SHA-256 was
`89d50f4e5d61be6079f8cda9455b8f0e7ca7d977136d343008f63a68362f4973`;
the AIR SHA-256 was
`e3661a41399068203716ea78daa66d4265e142c7d2fcdbc59e5e0b3990a83da3`.
Preflight HBM was 3,451-3,452 MiB with no visible process. Final HBM was 3,452
MiB for three samples, so relative recovery passed.

One DataFlow `LaunchKernel` record identifies the loaded custom kernel. It is
not a per-call counter because GraphPp loads one compiled model and executes it
repeatedly. The two sequenced, value-dependent AICore reports establish the two
executions. The earlier diagnostic run
`persistent-owner-v2-device-kv-update-20260815T190931Z` completed both updates
exactly but was rejected by the old verifier's incorrect two-record rule; its
result is not the authoritative pass.

## Boundary

This proves public lifetime and exact repeated mutation for one synthetic
FunctionPp-owned Device State Handle. It does not prove the target B4/K384
Paged-KV layout, update-to-attention ordering inside one graph, absence of
internal Device-to-Device copies, full Prefill/Decode correctness, or P5
performance. P5 remains Stopped / Unqualified and P6 remains closed.

The next gate is a target-layout update/read dependency probe: a
`DevicePagedKvUpdate` must update the declared B4/K384 slot and a downstream
Device reader must observe it through an explicit graph dependency, while the
Host-facing boundary remains metadata/report only.
