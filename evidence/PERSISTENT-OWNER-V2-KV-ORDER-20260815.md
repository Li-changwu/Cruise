# Controller-Aware Graph V2 target-layout ordering probe

The authoritative target-hardware run is
`persistent-owner-v2-kv-order-20260815T194310Z`, bound to clean source commit
`babcf33e05e0c47f2176abccaae5f0446cda1bf7`. Evidence integrity and the
workspace storage audit passed.

## Result

`V2-KV-ORDER` passed on physical NPU 0 of the pinned Ascend 910B2/CANN 9 stack.

- FunctionPp allocated one 3 MiB BF16 Device tensor with shape
  `[2, 12, 32, 128, 16]`, the target B4/K384 PA-NZ key/value layout. The same
  `FlowMsg` and Device data address remained stable across two GraphPp calls.
- The graph contained one `DevicePagedKvUpdate`, one `DevicePagedKvRead`, two
  ordinary `Data` inputs, and one compact output. The update ticket directly
  fed the reader. External `RefData`, `TensorMove`, and full-state outputs were
  absent. Graph SHA-256 was
  `f13a89e3cfa5e4ca42b42322cdaabe07c805e0d29da4285d407e9ed139633da7`.
- Each call updated all key/value features at logical positions 0, 127, 128,
  and 383. The Device reader checked all 4,096 updated BF16 elements after the
  dependency ticket. FunctionPp separately scanned all 1,572,864 state
  elements after each call; no missing, misplaced, or extra write was found.
- Ordinary Graph returned the exact 18-element Device report. Public GraphPp
  returned two exact sequence-dependent reports, preserved cross-call checksum
  continuity, and used one allocation for two graph calls.
- GraphPp Host cache input/output bytes were both zero. The Host sent 40 bytes
  of trigger metadata and received 384 bytes of summaries across two calls. No
  raw Device pointer was encoded in a tensor and no private `ModelPp` was used.
- Runtime metadata recorded one update and one reader kernel for each loaded
  ordinary Graph and GraphPp model, with `isNoNeedH2DCopy=1`. These records do
  not by themselves prove that every internal Device-to-Device copy is absent.

The isolated custom package SHA-256 was
`10555f37f30fd125212bc4825d0b91376c25ba7997317b40fa9bd0c43932aa27`;
the AIR SHA-256 was
`561cadbc3530ec8541bc42643ff8345e0f61acb19733f8d7af2430ff7737da20`.
Preflight HBM was 3,452-3,453 MiB with no visible process. Final HBM recovered
to 3,452 MiB across three stable samples.

The Torch-NPU exporter exited 139 during teardown after it had atomically
written `pass=true`, the AIR, and the graph-structure result. The runner's
documented teardown rule accepted that status only after verifying all three
files. Ordinary Graph and GraphPp then loaded and executed the resulting AIR
exactly, so teardown did not weaken the execution evidence.

## Boundary

This proves the target Paged-KV layout and explicit update-to-reader ordering
for one custom component graph. It does not prove real attention semantics,
absence of internal Device copies, whole-model Prefill/Decode export, service
correctness, or P5 performance. P5 remains Stopped / Unqualified and P6
remains closed.

The next gate must replace the synthetic reader with a real attention consumer
while retaining the passed Device State Handle ownership, B4/K384 layout, and
explicit update dependency. Only after that component is Graph/GraphPp exact
may whole-model graph-family export begin.
