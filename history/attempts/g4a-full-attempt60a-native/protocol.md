# G4a Attempt 60a native acceptance

Run the exported 28-layer AIR for four recurrent Host-loop steps on physical NPU 7.

Acceptance requires every step to satisfy `rtol=5e-3, atol=5e-3` for logits,
full Paged-KV and the newly written KV slot; greedy token and next position must
match; every unaddressed KV element must remain bitwise unchanged. The log must
show 112 ExactQk launches, 112 Bf16Barrier launches, and four launches for each
of the 113 `LinearTransposeX2*` nodes.

Native policy is frozen to `ge.exec.precision_mode=must_keep_origin_dtype`, with
`RESOURCE_CONFIG_PATH` and `ASCEND_CACHE_PATH` unset. This remains a Host-loop
G4a test and does not claim a device-resident epoch.
