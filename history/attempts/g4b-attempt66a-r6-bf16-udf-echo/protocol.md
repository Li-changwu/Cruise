# G4b Attempt 66a-r6: BF16 Host-to-Device-UDF echo

## Question

Does CANN 9.0 DataFlow preserve a `ge::DT_BF16` tensor across one Host Feed,
a Device UDF input/output, and one Host Fetch?

Attempt 66a-r5 proved that the Host creates `ge::Tensor` inputs with BF16 enum
27, but direct GraphPp execution receives `DT_UNDEFINED` for the first cache.
This experiment removes the decoder AIR and GraphPp entirely. The Device UDF
only validates a 16-element BF16 tensor and returns the same FlowMsg.

## Pass conditions

- Physical NPU 7 is idle before and after execution.
- The Device UDF observes `TensorDataType::DT_BF16`, 16 elements and 32 bytes.
- Fetch returns `ge::DT_BF16` and all 16 raw BF16 words are unchanged.
- The route uses one Feed and one Fetch.

Passing isolates the r5 failure to the GraphPp/model boundary. Failure before
the UDF runs establishes a CANN 9.0 Host-to-DataFlow BF16 transport limitation
and requires an explicitly typed byte-envelope experiment before G4b.
