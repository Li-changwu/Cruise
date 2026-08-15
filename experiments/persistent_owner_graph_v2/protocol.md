# Controller-Aware Graph V2 protocol

This experiment replaces only the Persistent Owner's compute-plane candidate.
The Device Control Plane remains authoritative for request lifecycle, Output
Credit, row and Device KV Lease ownership, phase transitions, and graph
selection. The Host may send request-boundary events and drain committed output;
it may not select a graph or issue a token-step command.

## First graph family

The first family is deliberately fixed to the formal concurrency-four workload:

- `Prefill-B4-L128` consumes all four 128-token prompts in one invocation.
- `Decode-B4` advances one token for four resident rows with a 384-token
  attention capacity.
- Both graphs use the same model-load-scoped Paged-KV state contract. A phase
  transition changes the selected closure, not KV ownership or location.

Graph implementations must retain merged QKV and gate/up projections, native
normalization and rotary operations, fused attention, and paged KV updates.
`SoftmaxV2` or `BatchMatMul` in the inspected attention path is a decomposed
fallback and fails the candidate before performance measurement.

The first KV experiment exported the public Torch-NPU
`npu_scatter_pa_kv_cache` path as `ScatterPaKvCache` with direct `RefData`
inputs. Its structure passed, but public GraphPp could not map external
`RefData` through the DataFlow boundary. That path is closed.

V2 now keeps shared state in a FunctionPp-owned Device State Handle. The public
handle is the retained `FlowMsg`; Cruise does not serialize its Device address.
An invoked GraphPp may receive that handle as an ordinary Device tensor input,
mutate declared slots with a graph-compatible custom AICore operator, and return
only compact state. The synthetic lifetime probe passed. Target Paged-KV layout
and update-to-attention ordering are still required before whole-model graphs.

V2 does not use the historical 5% idle-HBM line. New component runners require
no visible Device process, stable HBM samples, a broad retained-model safety
ceiling, and post-run recovery relative to the observed starting point.

## Ordered gates

1. `V2-CONTRACT` validates the checked-in manifest, exact extents, public
   invocation boundary, Device-owned selection, and forbidden Host/KV paths.
2. `V2-KV-REFDATA` records the closed external-`RefData` boundary; its structure
   passed but its Graph/GraphPp execution pair failed.
3. `V2-DEVICE-STATE-HANDLE` proves one FunctionPp allocation, stable `FlowMsg`
   identity/address, exact repeated GraphPp mutation, compact outputs, and zero
   Host cache I/O. The synthetic lifetime probe passed.
4. `V2-KV-ORDER` must prove the target B4/K384 layout and an explicit graph
   dependency from `DevicePagedKvUpdate` to a downstream Device reader. The
   bounded target-hardware probe passed on 2026-08-15.
5. `V2-EXPORT` exports each graph and records source, package, graph, external
   weight, and model identities. Contract validation alone does not satisfy it.
6. `V2-STRUCTURE` inspects real graph artifacts for required operators and
   projection/KV features and rejects decomposed attention.
7. `V2-GRAPH-PAIR` executes each identical artifact through ordinary Graph and
   public GraphPp, checking exact outputs and component time separately.
8. `V2-SHARED-KV` runs Prefill then Decode against one lease and proves the
   supported Device-handle/in-place boundary with actual copy evidence.
9. `V2-OWNER-MINI` lets the Device controller select both closures for one
   bounded request cohort. Only after it passes may a new ADR consider reopening
   a P5 Graph/Owner pair.

The passed ordering probe is component evidence only. The bounded public FIA
and PA-NZ consumer prerequisites later passed, but the one combined real-
attention gate failed before execution. TorchAir exported metadata at `Data`
index 2 and query at index 3, while the Graph config and C++ callers supplied
query at index 2 and metadata at index 3. GE rejected BF16 metadata for
`DevicePagedKvUpdate`; ordinary Graph launched no target kernel, and GraphPp
failed compilation before FunctionPp allocation or Feed. Under the frozen
stop rule, gates 5-9 do not advance and this path receives no corrected rerun.

No gate here changes ADR 0020. Until all component gates are backed by target
NPU evidence, the V2 state is `contract_defined`, P5 remains Stopped /
Unqualified, and the six-start matrix and P6 remain closed.

## Contract check

```bash
python experiments/persistent_owner_graph_v2/verify_graph_family.py \
  --manifest experiments/persistent_owner_graph_v2/graph_family.json
```

Adding two `--inspection` arguments changes the command into the candidate
gate. Each inspection must identify the artifact, exact extent, operator
counts, retained structural features, exact ordinary Graph and GraphPp outputs,
component timing, clean source and package identity, Device caller, and shared
KV proof. Prefill and Decode must use the same recorded source/package identity.
The emitted record deliberately limits its claim to those facts and never
reports performance qualification.
