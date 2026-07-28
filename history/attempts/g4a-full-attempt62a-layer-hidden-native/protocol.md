# G4a Attempt 62a native layer-hidden diagnostic

Run the exported 28-layer AIR for four recurrent Host-loop steps on physical NPU 7.

The graph adds one `[28,1,1,3584]` BF16 output containing the hidden state after
each decoder layer. The comparator reports the first exact mismatch and first
frozen-tolerance failure for each recurrent step. It also preserves all original
logits, KV, greedy, position and unaddressed-cache checks.

Native policy remains `must_keep_origin_dtype`, with `RESOURCE_CONFIG_PATH` and
`ASCEND_CACHE_PATH` unset. This diagnostic cannot pass G4a or enter G4b.
