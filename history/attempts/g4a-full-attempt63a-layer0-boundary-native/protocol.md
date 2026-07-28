# G4a Attempt 63a native layer-0 boundary diagnostic

Run the exported 28-layer AIR for four recurrent Host-loop steps on physical NPU 7.

The graph retains the per-layer hidden stack and adds 16 ordered layer-0
boundaries. The comparator reports the first exact mismatch and first frozen-
tolerance failure in this sequence for every recurrent step.

The 20 arrays already produced by Attempt 62a must remain elementwise
identical. Attempt 63a must contain exactly 84 arrays, with only the 64
`step{1..4}_layer0_{boundary}_bits` arrays added. A deterministic content hash
over the common arrays must also match.

Native policy remains `must_keep_origin_dtype`, with `RESOURCE_CONFIG_PATH` and
`ASCEND_CACHE_PATH` unset. This diagnostic cannot pass G4a or enter G4b.
