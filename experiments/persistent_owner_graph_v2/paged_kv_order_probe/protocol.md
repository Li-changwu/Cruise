# V2 target-layout Paged-KV ordering probe

This probe is the `V2-KV-ORDER` gate between the synthetic Device State Handle
lifetime proof and any whole-model Prefill/Decode integration. It retains one
FunctionPp-owned BF16 state tensor with shape `[2, 12, 32, 128, 16]`: key and
value caches, twelve physical pages, PA-NZ channels, 128 tokens per page, and
sixteen packed values. The twelve pages provide exactly B4/K384 capacity.

The Host sends only five INT32 values: a sequence and one logical token
position for each batch row. `DevicePagedKvUpdate` writes all 512 key and 512
value elements for each row at positions `[0, 127, 128, 383]`. Its compact
ticket is a required input to `DevicePagedKvRead`, which then checks all 4,096
updated elements. This ticket edge is the explicit in-graph update-to-reader
dependency. The graph returns only the reader report; it never returns the
state tensor.

FunctionPp invokes the same GraphPp closure twice with the same retained
`FlowMsg`. The first call writes sequence-one bit patterns into zero state. The
second call must observe those patterns before replacing them with
sequence-two patterns. After each call, the Device controller scans the full
3 MiB state tensor to reject any missing, misplaced, or extra write.

The gate requires:

1. one `DevicePagedKvUpdate` and one `DevicePagedKvRead`, with the update ticket
   directly feeding the reader;
2. one ordinary `Data` state input shared by both operators, no external
   `RefData`, no `TensorMove`, and no full-state graph output;
3. exact ordinary Graph execution and two exact public GraphPp executions;
4. target B4/K384 PA-NZ address mapping, exact full-state scans, and cross-call
   checksum continuity;
5. one FunctionPp state allocation, stable `FlowMsg` identity and Device data
   address, and zero Host cache input/output bytes.

This probe does not establish absence of internal Device-to-Device copies,
attention semantics, a complete Prefill or Decode graph, or P5 performance.
Any failed requirement stops this path; it must not be bypassed with external
`RefData`, Host cache I/O, private `ModelPp`, or an integer-encoded Device
pointer.

```bash
bash experiments/persistent_owner_graph_v2/paged_kv_order_probe/run_on_910b.sh
```
