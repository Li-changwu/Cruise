# Persistent Owner public FIA GraphPp stop record

Status: the single permitted P5 redesign cannot enter qualification on CANN
9.0 and Ascend 910B2. P5 is Stopped / Unqualified on this stack, and P6 remains
closed. This is a component-gate result, not a second P5 performance result.

## Public package identity

The isolated build used public `ops-transformer` tag `v9.0.0`, commit
`afe72144f9f2ac8441929035795db88a111b30c5`, target tiling key
`103000000000022220`, and variant
`FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da`.

| Artifact | SHA-256 |
|---|---|
| ordinary object | `525dd57621ab1a1259b9f9e281ccf6a57e40f711274bf60cccb24c113ba3af36` |
| ordinary metadata | `09ff06acef394d182f94fd3e9421a0d358825148fc6b7177e64df4543dc60da0` |
| relocatable object | `74a6d26132cd165c8ad3c0760ed80d29bd9c7ce191a3e281c1624d04ed6a6ab0` |
| relocatable metadata | `be1d5fb5569fc289e8d8501760fab510fdeeb5ede3d74e50f90d88b7de6789f4` |
| `binary_info_config.json` | `c55bf85edc5bb3adda6b221585b559327c4c61a168c688e733bc4937ba94a8ef` |
| `fused_infer_attention_score.json` | `433d397bce32d43372a323a96e9e27e17a73da305de0470fde39ee2fbbdd05fe` |
| `relocatable_kernel_info_config.json` | `581814c46253682fcd0fb44b71da5621e4430ff9b8a41fc2e7920b91b5f92b3a` |
| isolated R2 `.run` package | `b1addf2218bad0e968470b9e294bf5f5597b86432d25376592db470f55afded8` |

The project generator created the registration files. No registration JSON was
handwritten, the package was installed only under `/dev/shm`, and the system
CANN installation was not modified.

## Authoritative run

`persistent-decoder-p3-mini-fia-custom-r4-20260815-r1` used the public
`torchair.ge.custom_op` extension point to retain all 31 registered FIA input
slots. AIR inspection reconstructed the normalized `_input_name_key` and
`_input_name_value` mapping and passed the exact 0-30 slot contract. The AIR was
10,571 bytes with SHA-256
`d550a851388d419c99f5d4456d6ea39df6ac556bf7c0996c4e7c0792f2300f3c`.

Both execution modes failed before a kernel launch:

| Mode | Status | Exact output | Elapsed |
|---|---:|---|---:|
| ordinary Graph | `4294967295` | No | 5,243 ms |
| public GraphPp | `1343225857` | No | 5,297 ms |

Neither log contains the expected `3b093497...` static identity. Graph generated
`te_fusedinferattentionscore_8b383e9c...`; GraphPp generated
`te_fusedinferattentionscore_6897a7c...`. Both entered online precompile and
failed because the packaged source includes the unavailable relative header
`../../incre_flash_attention/op_kernel/incre_flash_attention_arch32.h`.

## Decision boundary

The earlier public ACL path proved exact FIA OM execution but could not expose
that OM to public GraphPp. This second supported path proved the 31-slot AIR and
package identities, but the runtime still did not select the static object.
There is no exact mini component pass, so the full P3 FIA graph, second
Graph/Owner pair, and six-start P5 matrix were not run.

The final check found no visible NPU process. NPU 0 retained 3,452/65,536 MiB
HBM (about 5.27%), so it also remained above the strict 5% formal readiness
threshold. No NPU reset was performed. ADR 0020 records P5 as Stopped /
Unqualified on this stack and keeps P6 closed.
