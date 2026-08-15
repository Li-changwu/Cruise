# Persistent Owner custom AICore GraphPp component probe

Status: the public CANN 9 GraphPp path can compile, load, and execute a minimal
custom AscendC AICore kernel on physical NPU 0. This closes a diagnostic
question only. It does not implement attention, qualify P5, open P6, or replace
the unchanged Graph/Owner performance gates.

## Authoritative run

`persistent-decoder-p3-custom-graphpp-20260815-r3` built the frozen Attempt 56r1
`Bf16Materialize` identity op from source in a PID-scoped `/dev/shm` project.
It installed the generated custom OPP package under that scratch tree, loaded
one exported AIR through ordinary GE Graph and public DataFlow GraphPp, and did
not modify the system CANN installation.

Both modes passed:

| Mode | GE status | Custom launches | Bitwise output | Elapsed |
|---|---:|---:|---|---:|
| Graph | 0 | 1 | Exact | 15,372 ms |
| GraphPp | 0 | 1 | Exact | 4,939 ms |

The input and both outputs had SHA-256
`227e5bb13b9ce9398eaa3ea61faa9c48c7575db602da429eba368f47d6c15bec`.
The kernel launch identity was `te_bf16materialize_`. Both modes released the
selected NPU process, final storage audit reported zero violations, and no raw
input/output `.bin`, compiler tree, or driver-log tree was retained in Git.

## Source identity

| Source | SHA-256 |
|---|---|
| `history/attempts/bf16-materialize-attempt56r1/bf16_materialize.cpp` | `facfaf2bab37376b9e7dfb9e0e60dd77357f0b8636115dae5a107e42937a2cfe` |
| `history/attempts/bf16-materialize-attempt56r1/bf16_materialize_def.cpp` | `78381a0852ef2540156bbd5a97e44efe77cbc757ff2fc0b0c26aee8f2202302e` |
| `history/attempts/bf16-materialize-attempt56r1/bf16_materialize_infershape.cpp` | `4c54a59e563038523b50f06d658e5eccbf66afecc4d0ced416f528eff5d4bd10` |
| `history/attempts/bf16-materialize-attempt56r1/bf16_materialize_tiling.cpp` | `9279c35f078e7b780c11e02a128cc2508a2a45f5da6d7850a83f7372f1603e75` |
| `history/attempts/bf16-materialize-attempt56r1/export_probe.py` | `d6a59c72e84b255375c7c52a908cd1136bf2aaa2ff1af98c2882ef8ea6da4d3f` |
| `experiments/persistent_decoder_p3/custom_graphpp_probe/prepare_probe_config.py` | `09ca7fcd30808f82236553c42056bfdbe9e581702d80da91c4860422b1c4b687` |
| `experiments/persistent_decoder_p3/custom_graphpp_probe/verify_probe.py` | `68fb2d2f3f68d788c2623de2ed28bcf17733e26ca190c12db63ea9251d66eef9` |
| `experiments/persistent_decoder_p3/custom_graphpp_probe/bf16_graphpp_probe.cpp` | `e446126dad5c8ff3cf096dfb47e8766687584593a68bb7877a5d92a4cdebf092` |
| `experiments/persistent_decoder_p3/custom_graphpp_probe/run_on_910b.sh` | `976404d8a9a7cdf63de98cacc50b199bbe5f1117b9270a0c68dfea0657d9c9ae` |

The compact verifier SHA-256 is
`22db22b158c5e0d48538899e046f15e9c23f66c9622fb160e461cb6d75e1b3d9`.

## Failure accounting and decision

Run r1 exported and executed the ordinary probe but exited with status 139
during TorchAir process teardown. Run r2 accepted that known post-export exit,
then exceeded the 64 MiB evidence budget because debug slog was retained; its
Graph and GraphPp processes were terminated with status 92 by the storage
guard. Run r3 validates complete result files after export, bounds driver-log
extraction, and keeps raw tensors in scratch. The r1/r2 failures were probe
infrastructure failures, not GraphPp kernel-loading observations.

Therefore `funcEntry=0` is not a general CANN 9 boundary for all custom AICore
kernels. The next P5 gate is a minimal, exact-output attention component over
the frozen B=4, K=384 shape. The installed FIA kernel source alone is not
enough to claim that its host tiling, tiling-data ABI, and OpDef can be
repackaged; those public dependencies must be established first.
