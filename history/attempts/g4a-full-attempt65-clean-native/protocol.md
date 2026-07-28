# G4a Attempt 65 native gate

Run the clean four-output AIR for four recurrent Host-loop decoder steps on
physical NPU 7 under `precision_mode=must_keep_origin_dtype`, with
`RESOURCE_CONFIG_PATH` and `ASCEND_CACHE_PATH` unset.

The run is accepted only when the semantic comparator and every runtime
mechanism check pass. This is the non-diagnostic G4a decision experiment; it
does not claim Device UDF residency or G4b.
