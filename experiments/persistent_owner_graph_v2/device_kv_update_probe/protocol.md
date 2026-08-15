# V2 Device-owned KV update lifetime probe

This probe asks whether one public FunctionPp-owned Device tensor can remain
alive across two public GraphPp invocations and be mutated exactly by a custom
AICore operator. It is the minimum safety gate before any target Paged-KV or
Decoder integration.

The installed public API accepts `FlowMsg` objects in `RunFlowModel` and states
that a tensor's lifetime follows its `FlowMsg`. It does not expose a documented
opaque Device-address wrapper for GraphPp. The probe therefore does not encode a
raw pointer in an integer tensor. FunctionPp retains one `FlowMsg` as the public
opaque buffer handle and passes that same object to the invoked GraphPp closure.

The Host sends two 16-byte metadata commands. FunctionPp allocates one 8 KiB
synthetic BF16 Device buffer, initializes two sentinels, and invokes the graph
twice. `DeviceKvSlotUpdate` changes one slot from zero to BF16 1.0, then from
BF16 1.0 to 2.0. The second call must observe the first call's value. FunctionPp
checks the complete synthetic buffer after each call and returns only a compact
summary and checksum; the Host never sends or receives the buffer.

The gate requires:

1. the AIR contains one `DeviceKvSlotUpdate`, ordinary `Data` inputs, no
   `RefData`, no `TensorMove`, and no full-buffer graph output;
2. ordinary Graph loads and runs the custom kernel with an exact small report;
3. public GraphPp invokes the kernel twice through a FunctionPp closure;
4. the same `FlowMsg` and data address remain stable, allocation count stays
   one, the second pre-update checksum equals the first post-update checksum,
   and both complete-buffer scans are exact;
5. GraphPp Host cache input/output bytes remain zero, and no raw-address ABI is
   used.

The runtime `LaunchKernel` record establishes the loaded custom-kernel identity,
not a per-invocation count: GraphPp loads the compiled model once and can execute
that model repeatedly without another record of the same kind. Execution count
is therefore established by the two exact AICore reports carrying distinct
sequence values and dependent before/after values.

This probe does not prove the target Paged-KV layout, safe scheduling against a
separate attention graph, absence of internal Device-to-Device copies, full
Prefill/Decode execution, or P5 performance. Any failed item stops this update
path instead of being bypassed with Host cache I/O, external `RefData`, private
`ModelPp`, or an undocumented integer pointer.

```bash
bash experiments/persistent_owner_graph_v2/device_kv_update_probe/run_on_910b.sh
```
