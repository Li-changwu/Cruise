---
status: accepted
---

# Retain shared state behind Device handles

The public GraphPp boundary cannot carry an external `RefData` cache, while the
public FunctionPp API gives a retained `FlowMsg` and its tensor one explicit
lifetime. Controller-Aware Graph V2 therefore keeps shared KV state in
FunctionPp-owned `FlowMsg` handles and passes the same handle to invoked GraphPp
closures; graphs may emit only compact results, not the full cache. Cruise will
not encode raw Device addresses in integer tensors or depend on private
`ModelPp`. The accepted synthetic lifetime probe proves exact repeated mutation
of one handle, but target Paged-KV layout, update-to-attention ordering, internal
Device-copy behavior, and full Prefill/Decode graphs remain separate gates.
