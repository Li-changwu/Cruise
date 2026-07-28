# G4a Attempt 64a native residual-defusion diagnostic

Run the Attempt 64a five-output full decoder for four recurrent Host-loop
steps on physical NPU 7 under `must_keep_origin_dtype`, with
`RESOURCE_CONFIG_PATH` and `ASCEND_CACHE_PATH` unset.

The immutable Attempt 62a result is the failure baseline. The mechanism is
supported only if the compiled launch metadata proves all 56 projection
residual joins are defused, all 56 Bf16Materialize kernels launch, and the
candidate passes logits, Paged-KV, greedy, position and per-layer tolerance
checks.

This diagnostic cannot enter G4b. A clean decoder graph without layer-hidden
outputs must independently pass before G4a is complete.
