# Controller-Aware Graph V2 KV attention stop record

The authoritative combined run is
`persistent-owner-v2-kv-attention-20260815-r1`, bound to clean source commit
`864fb94092edfb857fd3d399a28b25ced05daec1`. The combined path is stopped.
This is a failed component gate, not a P5 performance result.

## Passed prerequisites

- The two custom AICore operators and public ops-transformer v9.0.0 FIA OPP
  configured, built, packaged, and installed in isolated `/dev/shm` scratch.
  The system CANN/OPP installation was not modified.
- AIR export completed before the accepted exporter teardown status 139. The
  AIR contained six `Data` nodes, one `DevicePagedKvUpdate`, one
  `DeviceQueryAfterKvUpdate`, one `FusedInferAttentionScore`, and one compact
  `NetOutput`.
- Structure inspection proved that update and FIA consumed the same key/value
  nodes, the ticket ordered the query before FIA, and no `RefData`,
  `TensorMove`, or full-KV output existed.
- The AIR SHA-256 was
  `1e3facd1eed417daeccd345fd479d256c676c9c465575a55cae3f069fca0784f`;
  the graph-text SHA-256 was
  `e49ec3f5ae3bd36e781121b4bfc7722f825e983ba018f200fc244464f1a1cb03`;
  the isolated custom package SHA-256 was
  `81159df31e74d9c1c54d914659d5f13b799f5bc719fd3fc4c7b66416fb699334`.

## Deterministic first failure

TorchAir reordered the exported `Data` indices according to graph use:

1. key cache, BF16;
2. value cache, BF16;
3. metadata, INT32;
4. query, BF16;
5. mask, BOOL;
6. block table, INT32.

The checked-in Graph compile config and both C++ callers still used the Python
function order: key, value, query, metadata, mask, block table. GE therefore
bound the BF16 query descriptor at input index 2 to
`DevicePagedKvUpdate.metadata`, whose public operator contract requires
INT32. Engine selection failed deterministically with error `1343242282` and
the message that BF16 metadata is unsupported.

Ordinary Graph loaded a valid AIR and prepared Device-placed inputs, but
`RunGraph` failed before any target kernel launch. Public GraphPp compiled the
FunctionPp library, then failed graph compilation with status `1343225857`
before the first Feed. Its FunctionPp controller therefore made zero
allocations and zero graph calls. No buffer-lifetime, exact update, attention,
or dual-route execution claim can be made.

## Decision boundary

The gate required every public API, input/lifetime, exactness, no-full-KV-I/O,
and dual-route condition to pass in one combined run. The input-index contract
was not satisfied, so no corrected rerun, full Decoder integration, P5 pair,
or six-start matrix is authorized on this path. P5 remains Stopped /
Unqualified and P6 remains closed.

The run exited 1, retained bounded failure logs, released NPU 0, and removed
its scratch. Final NPU 0 HBM was 3,452 MiB with no visible process, and the
`/dev/shm` audit reported zero violations. No NPU reset was performed.
