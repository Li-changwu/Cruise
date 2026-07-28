# Remote Storage Policy - 2026-07-26

## Scope

This policy covers the remote G4 experiment state used by Attempt 69e and the
later performance/recovery gates. Scientific evidence is the smallest
immutable set needed to verify a claim. Rebuildable caches, duplicate external
weights, build trees, and verbose runtime logs are not evidence by themselves.

## Closed-G4 Dependency Closure

G4 is closed by Attempt 70b-r3-r1. The retained remote closure is now source,
compact evidence, accepted references, the AIR graph, and required custom
operators:

- `/root/ascend-control-g4-20260723/attempt69e-src`
- `/root/ascend-control-g4-20260723/export-attempt69c-r2-b4/qwen_b4_decoder_step_attempt69c_r2.air`
- `/root/ascend-control-g4-20260723/raw-attempt69b-b4-eager/outputs/attempt69b-b4-eager-reference.npz`
- `/root/ascend-control-g4-20260723/raw-attempt69d-r1-b4-native/attempt69d-r1-acceptance.json`
- `/root/ascend-control-g4-20260723/install-attempt69a-b4-barrier`
- `/root/ascend-control-g4-20260723/install-attempt56r1`
- `/root/ascend-control-g2g-20260719/install-attempt47`
- `/root/ascend-control-g2g-20260719/numa_config.physical7.json`
- CANN 9.0.0 and the `vllm-hust-dev` Conda environment

The accepted result JSON, status, integrity logs, comparison outputs, source,
and protocol files remain evidence. A future rerun must rematerialize external
weights into guarded `/dev/shm` scratch; there is no durable root-disk weight
copy after closure.

## Removed On 2026-07-26

The following nine unreferenced full-weight copies were removed:

- `export-attempt53k`
- `export-attempt60a-r2`
- `export-attempt61a`
- `export-attempt62a`
- `export-attempt63a`
- `export-attempt64a`
- `export-attempt65`
- `external-weights-attempt66b-r4`
- `external-weights-attempt68a`

The following old export/debug artifacts were also removed:

- `export-attempt55a`, `export-attempt55d`, `export-attempt57a`,
  `export-attempt58a`, and `export-attempt59a`
- every historical `raw-*/native.stdout.log` under the G4 root
- all of `/root/ascend-control-g2e-20260718`
- every directory under `/root/ascend-control-g2g-20260719` except
  `install-attempt47`

The accepted Attempt 69d-r1 debug stdout had already been copied to the local
archive and verified with SHA256
`aaa055b5efca0c84d458b78eb3c74b9d2dfb868183455abb1d41376599d9b152`
before its remote copy was removed.

## Removed After G4 Closure On 2026-07-27

After the completion verifier passed and remote/local evidence hashes matched,
`prune_closed_g4_storage.sh` verified the completion JSON, both evidence
manifests, exact target paths, symbolic-link references, and open file
descriptors. It then removed:

- `export-attempt67b-b2` and `export-attempt69c-b4`: two hard-link views of one
  rebuildable weight family, releasing 15,278,657,536 root-disk bytes;
- `/dev/shm/ascend-control-g4-20260726/attempt69e-r5-b4-resident-epoch`: a
  pinned but already archived scratch tree, releasing 16,756,948,992 bytes.

The deletion is not recoverable as a filesystem operation, but both targets
are rebuildable. Scientific evidence, AIR, references, source, and hashes were
not deleted.

## Retained But Not In The 69e Runtime Closure

- `/root/ascend-control-g0g1-20260717`: retains 1.4 GB of original
  motivation/baseline profiles for later paper-evidence selection; it is not a
  runtime dependency and remains below the 2 GiB manifest threshold.
- `/root/.triton`: rebuildable, but shared with other Triton work and expensive
  to redownload; do not remove merely to make room for G4.
- `/root/ascend/log`: runtime diagnostics; rotate after a gate closes, not
  during an unresolved failure.
- `/root/.vscode-server`: currently active remote-development service.
- `/root/triton-ascend-hust`: source repository, not a cache.
- `/root/miniconda3`: contains the required experiment environment.

## Measured Effect

- Root free space: 4.7 GB before dependency-driven cleanup, 164 GB after.
- G4 root: 177 GB before cleanup, 23 GB after.
- G2g root: 1.9 GB before pruning, 6.5 MB after.
- G2e root: 4.0 GB before cleanup, removed.

Post-cleanup integrity checks preserved:

- relocated B=4 AIR SHA256:
  `263b2acf291e13f6a84042ded53c8dccabb1fa847dcdcbbbe0ece418610ad1e3`
- B=4 eager reference SHA256:
  `a7d65e455a77a561352a8f3796d94ec86e1e429ebe942feacd5b14013123fdd8`
- B=4 native acceptance SHA256:
  `72e73a176097d9368171b866a384bf70c33616adcdd4cfccc53a015db0681605`
- `export-attempt69c-b4` still contains all external-weight files, and its
  `embedding` remains hard-linked to the provenance base in
  `export-attempt67b-b2`.

The post-closure cleanup on 2026-07-27 further changed the live state to:

- root free space: 139,021,029,376 bytes;
- G4 persistent tree: 3,852,775,424 bytes;
- all `ascend-control-*` roots: 5,799,280,640 bytes;
- G4 `/dev/shm` scratch: 20,267,008 bytes.

The earlier hard-link statement is retained above only as the 2026-07-26
measurement. Both weight views were removed after the performance gate closed.

## Rule For Future Attempts

Place transient GraphPp external weights and GE caches in `/dev/shm`. Keep only
one durable weight set per live graph family. After a gate closes, retain its
source, protocol, result, status, hashes, compact comparison outputs, and any
irreplaceable profiler data; remove build directories, duplicate graph
packages, and verbose success-path logs.

All G4 runs starting with Attempt 69e-r1 must use the checked-in storage guard.
It rejects unsafe launches before creating evidence or scratch targets, keeps
stdout/stderr evidence bounded to a 32 MiB head and 32 MiB tail, and monitors
filesystem budgets for the lifetime of every heavy command.

The limits were tightened after the 2026-07-26 overlay audit. New attempts must
keep at least 100 GiB free on root and 128 GiB free in `/dev/shm`; the G4
persistent tree is capped at 24 GiB, each evidence directory at 512 MiB, and
each scratch tree at 64 GiB. A preflight snapshot records the size of every
top-level persistent artifact. More than 64 MiB of unexpected growth outside
the current evidence directory terminates the heavy command, even if the
24 GiB aggregate cap has not yet been crossed.

After a successful result, final NPU-idle check, and evidence hash validation,
the attempt removes its own marker-protected scratch tree. Failed attempts keep
their scratch for diagnosis; the next run remains blocked if any filesystem
budget is crossed. The guard never deletes scientific evidence, shared caches,
or an unmarked scratch tree.

The root filesystem is an overlay. On 2026-07-26, `df` reported about 3.66 TB
used while `du -x /` could account for only about 73.0 GB inside this container.
The remaining backing-store use is outside the container's visible namespace,
so project cleanup cannot reclaim or attribute it. Every new attempt must run
`storage-control/audit_storage.sh`; a low root reserve is a hard block and must
be escalated to the server administrator rather than worked around by moving
more files inside the same overlay.

The audit now also covers `/root` as a whole. `/root` is capped at 48 GiB, all
`ascend-control-*` roots together are capped at 16 GiB, and an individual
top-level root directory may not exceed 2 GiB unless it is listed in
`root-retention-manifest.tsv` with a byte ceiling. This catches new experiment
roots, cache expansion, and many-small-project accumulation instead of only
watching the current G4 directory.

An empty NPU process list is also insufficient by itself. Attempt 70a observed
no process immediately after a shared job exited, but then failed to load the
flow model with HBM OOM. New launches require HBM usage at or below 5% for
three consecutive five-second samples in addition to an empty process list.

## Continuous Enforcement

Because this container has neither cron nor a running systemd instance, an
idempotent bootstrap in `/root/.bashrc` starts a watchdog supervisor under a
singleton `flock`. The supervisor restarts `storage_watchdog.sh` after an
unexpected process exit. It audits the fast-changing budgets once per minute
and keeps only the current snapshot plus a bounded transition history. Every
SSH login repairs the supervisor if the container has restarted.

The watchdog creates `storage-control/state/BLOCK` whenever root reserve,
shared-memory reserve, `/root` aggregate size, all control-root aggregate size,
G4 project size, aggregate G4 scratch size, an unregistered large artifact, or
Ascend log size crosses its limit. Aggregate G4 scratch is capped at 96 GiB.
The per-project manifest governs G4 children; the root manifest independently
governs `.triton`, `.vscode-server`, Conda, source trees, and experiment roots.

Once per hour the watchdog may rotate only closed files under
`/root/ascend/log` when that tree exceeds 2 GiB, deleting files older than
three days only until the tree is below 1 GiB. Creating
`/root/ascend/log/.storage-retention-pause` disables that deletion while a
failure is unresolved. It may also remove only finalized, unpinned G4 scratch
with all-zero statuses, a passing result, verified evidence hashes, no open
files, and an elapsed grace period. Failed or unverifiable scratch remains
manual-review state.

The child explicitly does not inherit the bootstrap lock; this closes the
one-shot restart bug found in the first deployment. The supervisor covers
watchdog failure while the container remains alive. Starting it before any
SSH login after a container restart requires host-administrator support.

The supported launch path is:

```bash
/root/ascend-control-g4-20260723/storage-control/run_guarded_attempt.sh \
  /root/ascend-control-g4-20260723/attemptNN-src
```

This performs a fresh audit and static guard validation before executing the
attempt. Validation pins the SHA256 of `storage_guard.sh` and
`bounded_log.py`, so retaining only the expected call sites while weakening
the implementation is rejected. Running an attempt source directly is
outside the reproducible protocol and must not be used for formal evidence.
