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

The first implementation target for KV updates is the public Torch-NPU
`npu_scatter_pa_kv_cache` path, exported as `ScatterPaKvCache`. TorchAIR contains
a reference-input optimization that can replace copied cache inputs with
`RefData`. V2 requires one update per layer, a proven KV `RefData` alias, and no
KV-path `TensorMove`. The presence of this software path is design evidence
only; the actual exported graph must establish that the optimization fired and
that public GraphPp can load and execute the resulting artifact.

## Ordered gates

1. `V2-CONTRACT` validates the checked-in manifest, exact extents, public
   invocation boundary, Device-owned selection, and forbidden Host/KV paths.
2. `V2-EXPORT` exports each graph and records source, package, graph, external
   weight, and model identities. Contract validation alone does not satisfy it.
3. `V2-STRUCTURE` inspects real graph artifacts for required operators and
   projection/KV features and rejects decomposed attention.
4. `V2-GRAPH-PAIR` executes each identical artifact through ordinary Graph and
   public GraphPp, checking exact outputs and component time separately.
5. `V2-SHARED-KV` runs Prefill then Decode against one lease and proves the
   supported Device alias/in-place boundary with actual copy evidence.
6. `V2-OWNER-MINI` lets the Device controller select both closures for one
   bounded request cohort. Only after it passes may a new ADR consider reopening
   a P5 Graph/Owner pair.

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
