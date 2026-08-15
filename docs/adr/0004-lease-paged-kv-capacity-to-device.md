# Lease Paged-KV capacity to Device

The Host retains capacity admission and grants each admitted request a
multi-page Device KV Lease covering its declared prompt and output budget. The
NPU owns position, block-cursor, and KV progression within that lease, with no
per-token Host allocation. The first Persistent path does not extend a live
lease: admission waits or fails unless the complete prompt-plus-output budget is
available, and Device execution stops at EOS, output budget, or lease end,
whichever comes first. Lease return occurs at a bounded completion,
cancellation, or fault boundary. This replaces the fixed single-block row model
and allows the accepted 128-token prompt and 256-token output target without
reintroducing per-token Host control.
