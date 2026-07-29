# Runtime Configuration

Cruise accepts one strict JSON configuration. Start from
`config/cruise.example.json` and keep the machine-specific copy outside the Git
checkout, for example at `/etc/cruise/cruise.json`. Unknown fields, missing
fields, invalid integers and unknown compatibility profiles are rejected.

## Top-Level Fields

| Field | Purpose |
|---|---|
| `schema_version` | Must equal the packaged runtime configuration version |
| `compatibility_profile` | Exact software/hardware profile checked by `doctor` |
| `device_id` | Physical NPU exposed to the child process |
| `cann_set_env` | CANN environment script sourced by `cruise run` |
| `custom_opp_vendors` | Ordered relocatable OPP vendor roots |
| `runtime` | Epoch, timeout, scratch and storage limits |
| `assets` | Native server, AIR, configs, weights and model metadata |
| `integrity` | Expected hashes and external-weight count/bytes |

Every OPP vendor must contain `op_impl`, `op_proto` and `op_api/lib`. Cruise
builds `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH` from these roots; generated
`set_env.bash` files with stale install-time paths are not used.

## Runtime Fields

`scratch_root` must be a dedicated child of `/dev/shm`, never `/dev/shm`
itself. `external_weights` must also be below `/dev/shm` but outside the managed
scratch root so normal cleanup cannot delete model assets.

`max_steps` is one of 1, 2, 4 or 8 and cannot exceed `logical_capacity`.
`minimum_scratch_free_bytes` is checked before the child starts.
`persistent_output_limit_bytes` records the release retention bound; the
current runner emits no persistent runtime tree and cleans its per-run scratch
unconditionally.

## Asset Integrity

Normal validation hashes AIR, graph configuration, tiling, model metadata and
the external-weight manifest. It also checks external-weight file count and
aggregate bytes. Deep validation additionally hashes every weight described by
the manifest:

```bash
cruise config validate /etc/cruise/cruise.json
cruise config validate /etc/cruise/cruise.json --check-paths
cruise config validate /etc/cruise/cruise.json --deep
```

The function configuration is semantic rather than content-hash checked because
its `workspace` is deployment-specific. It must select
`g4c_b4_resident_epoch`, declare exactly 8 inputs and 2 outputs, and reference
an existing controller workspace.

## Environment Boundary

`cruise run` derives the legacy `VLLM_ASCEND_RESIDENT_EPOCH_*` variables from
the validated configuration. They remain an internal bridge for the current
Python/backend implementation and are not the user configuration contract.
The command does not accept arbitrary environment keys from JSON, so a config
file cannot silently inject preload libraries, Python paths or credentials.

