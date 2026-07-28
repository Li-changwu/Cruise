# Root Storage Audit - 2026-07-26

## Verdict

The earlier 189 GB G4 expansion was caused by duplicate graph weight packages,
raw outputs, build trees, and unbounded logs. That project-local issue has been
reduced and is now guarded. The server's current 97% root usage is a separate
host-level overlay issue: after post-closure cleanup only 51.8 GB is visible
from this container, while the backing filesystem reports about 3.64 TB used.

## Current Measurements

| Metric | Bytes | Approximate size |
|---|---:|---:|
| Root filesystem used (`df`) | 3,640,263,774,208 | 3.39 TiB |
| Namespace-visible root (`du -x /`) | 51,765,010,432 | 48.2 GiB |
| Backing store not visible here | 3,588,498,763,776 | 3.26 TiB |
| Root filesystem available | 139,021,029,376 | 129.5 GiB |
| `/root` visible tree | 32,709,836,800 | 30.5 GiB |
| All `ascend-control-*` roots | 5,799,280,640 | 5.4 GiB |
| G4 persistent tree | 3,852,775,424 | 3.6 GiB |
| `/root/ascend/log` | 37,584,896 | 35.8 MiB |
| G4 scratch in `/dev/shm` | 20,267,008 | 19.3 MiB |

The invisible difference can include host files, other overlay layers or
containers, and storage not mounted into this container. It cannot be safely
attributed or deleted through this SSH namespace.

## Retention Decision

- Recovery and stable performance are closed. The single 15 GB durable B=4
  weight family and its hard-link view were removed after hash verification.
- Keep the 543 MB B=4 eager reference, accepted native result, required custom
  operators, compact evidence, sources, and protocols.
- Keep `/root/.triton`, `/root/.vscode-server`, `/root/triton-ascend-hust`, and
  `/root/miniconda3`; they are shared caches, active tooling, source, and the
  required environment rather than disposable G4 output.
- Keep the 1.4 GB G0/G1 profiles for later paper-evidence selection. They are
  not a runtime dependency and remain below the 2 GiB manifest threshold.
- Old G4 raw tensors and global Ascend logs are not part of the current runtime
  dependency closure. On 2026-07-26, 47 old raw files (4,276,883,089 bytes) and
  6,448 logs older than three days (1,723,281,708 bytes) were deleted. The B=4
  eager reference, native acceptance, AIR, and accepted result hashes were
  checked before and after deletion.

The four experiment roots from the original alert now have these decisions:

| Directory | Current state | Decision |
|---|---:|---|
| `/root/ascend-control-g4-20260723` | 3.6 GiB | Keep compact source, AIR, references, custom operators, and evidence. Durable external weights are removed; future reruns must rematerialize them in guarded `/dev/shm`. |
| `/root/ascend-control-g2e-20260718` | absent | Deleted; it was not in the live dependency closure. |
| `/root/ascend-control-g2g-20260719` | 6.5 MiB | Keep `install-attempt47` and NUMA configuration; both are live Device UDF dependencies. |
| `/root/ascend-control-g0g1-20260717` | 1.4 GiB | Keep as original motivation/profile evidence until paper-evidence selection; it is not a runtime dependency. |

`/root/.triton`, `/root/.vscode-server`, `/root/triton-ascend-hust`, and
`/root/miniconda3` are not copies of this experiment. They remain shared cache,
remote IDE, source, and environment state and must not be bulk-deleted by the
G4 retention job.

## Prevention Contract

New attempts are blocked below 100 GiB root free or 128 GiB `/dev/shm` free.
They are also blocked above 24 GiB persistent G4 use. Heavy output must live in
marker-protected `/dev/shm` scratch, persistent evidence is capped at 512 MiB,
and unexpected persistent growth above 64 MiB terminates the running command.
Successful attempts remove their scratch only after evidence integrity passes.
All G4 scratch trees together are capped at 96 GiB. Any persistent top-level
directory above 2 GiB is rejected unless it appears in
`storage-control/retention-manifest.tsv` with its own byte ceiling.

The containing namespace is now budgeted too: `/root` is capped at 48 GiB,
all `ascend-control-*` directories together at 16 GiB, and any `/root` child
above 2 GiB must be approved in `root-retention-manifest.tsv`. Listed caches
and tools have individual ceilings; crossing one blocks formal runs without
automatically deleting shared state.

## Deployed Enforcement

`storage-control/storage_watchdog.sh` is now a singleton background process
under `storage_watchdog_supervisor.sh`.
It refreshes `storage-control/state/current.tsv` every minute and writes a
`BLOCK` marker on any budget violation. A managed `/root/.bashrc` block runs
`ensure_storage_watchdog.sh`, so the next SSH login repairs the process after
termination or container-local process loss. The restart path was tested by
terminating watchdog PID 2614020 and observing supervisor PID 2682812 launch
replacement watchdog PID 2683698 without another SSH bootstrap. The deployment
also fixed an inherited startup-lock descriptor that had prevented the old
`ensure` path from running more than once.

The watchdog performs no project deletion. Once per hour it checks Ascend
logs; only when `/root/ascend/log` exceeds 2 GiB may it delete closed files
older than three days, stopping below 1 GiB. The marker
`/root/ascend/log/.storage-retention-pause` disables rotation for unresolved
failures. The rotation behavior, pause behavior, fast audit, syntax, and
watchdog state transition were tested in isolated `/dev/shm` trees.

Once per hour it also considers marker-protected G4 scratch for deletion. A
tree is removable only when it is finalized, older than the grace period, not
pinned, has no open file, every recorded status is zero, contains a passing
result, and its persistent evidence passes SHA256 verification. Failed or
unverified scratch is reported but never deleted automatically. The 70b-r1
failure evidence was copied locally with 25/25 hashes matching and its H1 TSV
hash verified before its exact 417 MB scratch tree was removed.

Formal attempts must be launched through
`storage-control/run_guarded_attempt.sh`. The launcher runs a fresh storage
audit and validates fixed thresholds, scratch placement, bounded logs, exit
finalization, and pinned SHA256 values for the guard implementation. This
does not control the roughly 3.26 TiB backing-store usage outside the
container namespace; crossing the 100 GiB reserve for that reason remains an
administrator escalation and a hard experiment block.

After G4 closure, `prune_closed_g4_storage.sh` removed only three exact,
predeclared paths after checking completion/evidence hashes, JSON pass status,
open descriptors, symbolic-link references, and scratch provenance. It
released 15,278,657,536 root bytes and 16,756,948,992 shared-memory bytes. The
post-cleanup storage audit and isolated storage-control self-test both pass.
