# Storage Prevention Closure - 2026-07-27

## Verdict

The previous root-disk expansion is now addressed at three levels: heavy
attempt output is forced into bounded `/dev/shm` scratch, closed artifacts have
a hash-gated deletion path, and the watchdog audits the whole `/root`
namespace rather than only the active G4 project.

## Cleanup Evidence

`prune_closed_g4_storage.sh --apply` accepted only three exact paths after the
completion JSON, Attempt 70b-r3-r1 evidence, Attempt 69e-r5 evidence, scratch
provenance, open-file, and symbolic-link checks passed.

| Storage class | Deleted bytes | Disposition |
|---|---:|---|
| Root external-weight hard-link family | 15,278,657,536 | Rebuildable; rematerialize only in guarded scratch |
| Pinned closed Attempt 69e-r5 scratch | 16,756,948,992 | Evidence already archived and hash-verified |

The deletion is not filesystem-recoverable. No scientific evidence, source,
AIR, eager/native reference, result JSON, status file, or integrity manifest
was deleted.

## Enforced Budgets

| Scope | Hard limit |
|---|---:|
| Root free-space reserve | 100 GiB minimum |
| `/dev/shm` free-space reserve | 128 GiB minimum |
| `/root` visible aggregate | 48 GiB maximum |
| All `/root/ascend-control-*` roots | 16 GiB maximum |
| Closed G4 root | 6 GiB maximum through root manifest |
| G4 persistent attempt tree | 24 GiB maximum |
| All G4 scratch | 96 GiB maximum |
| One attempt scratch | 64 GiB maximum |
| One evidence directory | 512 MiB maximum |
| Unlisted persistent child | 2 GiB maximum |
| Ascend logs | rotate above 2 GiB toward 1 GiB |

`.triton`, `.vscode-server`, Miniconda, and large source trees are retained but
have explicit individual ceilings. The watchdog blocks new formal attempts on
growth; it never bulk-deletes shared caches or unverifiable evidence.

## Final Audit

- storage audit: `PASS`, reason `none`;
- root available: 139,021,029,376 bytes;
- `/root` visible: 32,709,836,800 bytes;
- all control roots: 5,799,280,640 bytes;
- G4 persistent: 3,852,775,424 bytes;
- G4 scratch: 20,267,008 bytes;
- unapproved or oversized root directory: none;
- NPU 7: no process, HBM 5%, AI Core 0%;
- storage-control self-test: `PASS`.

The remaining approximately 3.59 TB backing-store usage is outside this
container's visible namespace. The 100 GiB reserve converts future host-side
growth into a hard experiment block; reclaiming or attributing that invisible
space requires the server administrator.
