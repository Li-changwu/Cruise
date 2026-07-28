# Repository and Storage Policy

Cruise keeps reproducible source and compact textual protocols in Git. It does
not use Git as storage for models, compiler products, caches, profiler exports,
or raw experiment directories.

## Included

- Python, C/C++, shell, CMake, configuration, tests, and protocols;
- the active Attempt 74 implementation at the repository root;
- source-only historical snapshots under `history/attempts/`;
- compact human-readable status documents and checksums.

The initial import copied 991 source/configuration/documentation files from the
research workspace, including 104 historical attempt directories. Their total
size was 5.46 MiB before repository documentation was added.

## Excluded

- model weights and token/data binary files;
- AIR/OM packages, object files, shared libraries, and kernel caches;
- `msprof`, CANN, GE, runtime-copy, and application logs;
- NumPy arrays, raw CSV exports, crash dumps, sockets, and scratch trees;
- the original `archive/` and `results/` directories;
- Python, Triton, IDE, Conda, and build caches;
- host-specific credentials and SSH aliases.

The `.gitignore` blocks common forms of these artifacts. Run
`python scripts/audit_repository.py` before every push. The audit rejects
forbidden artifact types/directories, files above 5 MiB, a repository payload
above 50 MiB, common private-key/token formats, and the original SSH alias.

## Server-Side Experiment Storage

Formal NPU experiments must use the storage guard in `storage_guard/` and a
marker-protected scratch root, normally on `/dev/shm`. Before loading a model,
the guard checks root free space, aggregate scratch usage, per-directory
retention, NPU process/HBM readiness, and output bounds. Generated artifacts
stay in scratch; only a compact evidence set remains on persistent storage.

Do not create a new persistent `ascend-control-*` tree by copying a previous
experiment. Start from the tracked source, allocate a guarded scratch root,
and retain only the protocol, result summary, status, hashes, and bounded
diagnostic excerpts needed to support a claim.

## Historical Snapshot Rule

Files below `history/attempts/` are provenance snapshots. Do not silently fix
or modernize them. New work belongs at the repository root or in a new
explicitly named experiment directory; corrections to historical claims must
be recorded in a new document that references the original snapshot.
