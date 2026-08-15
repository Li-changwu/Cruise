# V2 KV Attention Safety Probe

This is the bounded gate before any full Decoder or P5 qualification work. It
combines the already-proven Device State Handle, B4/K384 PA-NZ update ordering,
and public FIA GraphPp path in one graph.

FunctionPp owns separate long-lived key and value buffers with shape
`[12, 4, 8, 128, 16]`. Host sends only a 20-byte sequence/slot metadata tensor.
`DevicePagedKvUpdate` writes four declared slots in place and emits a 9-int
ticket. `DeviceQueryAfterKvUpdate` consumes that ticket and forwards only the
28 KiB query, so FIA has an explicit graph dependency on the completed update.
FIA reads the same key/value Data nodes directly. A mask exposes only the four
updated logical slots, making the attention output bitwise equal to the value
written for that sequence.

The gate passes only when:

1. the public custom-op and FIA JIT toolchains build in isolated scratch;
2. AIR contains one update, one query-order primitive, and one FIA, with no
   `RefData`, `TensorMove`, or full-KV graph output;
3. ordinary Graph uses Device-placed KV inputs, returns exact compact output,
   and performs no full-KV Host input/output;
4. FunctionPp allocates the two KV buffers once, preserves both FlowMsg and
   Device address identity across two GraphPp calls, and full-scans the state;
5. each call observes exact slot-only state and an attention output equal to
   that call's newly written BF16 value;
6. Host KV input/output is 0/0 bytes and no raw address integer ABI, external
   `RefData`, private `ModelPp`, or Host `aclmdlExecute` is used.

Any failure stops this combined path. A pass remains a component result: it
does not prove zero internal Device-to-Device copies, full-model correctness,
service behavior, performance, or P5 qualification.
