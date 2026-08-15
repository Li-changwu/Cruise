# Minimal Custom AICore GraphPp Probe

## Question

Can the installed public CANN 9 GraphPp path compile, load, and execute a
minimal custom AscendC AICore kernel, or is `funcEntry=0` a general online
custom-kernel loading boundary on this stack?

## Frozen Mechanism

The probe reuses the repository-frozen Attempt 56r1 `Bf16Materialize` source.
It is a single AIV identity kernel over contiguous BF16 `[1, 1, 18944]`. The
runner copies the installed CANN custom-op template into PID-scoped `/dev/shm`,
builds and installs the package there, exports a one-node AIR, and executes the
same AIR through ordinary GE Graph and public DataFlow GraphPp.

The system CANN installation is not modified. Its root-only `opp/vendors`
directory is excluded during compilation with a user-owned OPP proxy that
links only the public built-in OPP directories required by `opc`. The generated
custom package is loaded through its public `set_env.bash` entrypoint.

## Pass Rule

Both Graph and GraphPp must:

- load a valid AIR and return success from their execution path;
- produce output that is bitwise identical to the same frozen input;
- emit at least one `te_bf16materialize_` kernel launch in the bounded log;
- release the selected NPU process before the next mode and after completion.

A Graph pass with a GraphPp `funcEntry=0` failure establishes a DataFlow-only
custom-kernel loading boundary. Failure in both modes is a custom-op build or
ordinary GE integration failure and does not isolate GraphPp.

## Claim Boundary

This is a minimal compute-plane component gate only. It does not implement
FusedInferAttentionScore, attention semantics, the Persistent Device Model
Owner, P5 performance, or the six-start qualification matrix.

## Accepted Result

Run `persistent-decoder-p3-custom-graphpp-20260815-r3` passed both modes with
one `te_bf16materialize_` launch and bitwise-exact output per mode. The compact
record is `evidence/PERSISTENT-OWNER-CUSTOM-GRAPHPP-20260815.md`. This result
rejects a general custom-kernel loading boundary and advances only to a
minimal exact-output attention component gate.
