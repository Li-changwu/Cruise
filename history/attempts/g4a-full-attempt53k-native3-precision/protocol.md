# G4a Attempt 53k Native3 Precision-Isolation Gate

Date frozen: 2026-07-23

## Claim

Attempt 53k native2 proved that the exported complete-decoder AIR can execute
four recurrent steps, but every logits tensor contained non-finite values and
the written Paged-KV slots diverged from the frozen eager reference. Native2
removed three runtime settings at once after native1 failed while copying the
15.2 GB external-weight image.

Native3 changes exactly one runtime variable relative to native2: it restores
`ge.exec.precision_mode=must_keep_origin_dtype`. `RESOURCE_CONFIG_PATH` remains
unset and the GE compiler cache remains disabled. Runtime log level 0 is used
to preserve actual custom-kernel launch records.

## Frozen Inputs And Outputs

- AIR and 342 external files: immutable Attempt 53k export.
- Reference: immutable Attempt 53k eager reference.
- Runtime device: idle physical NPU 7.
- Recurrence: four Host-driven complete decoder steps, each consuming the
  preceding native K/V and position outputs.

## Pass Conditions

For all four steps:

- logits and complete K/V are within `rtol=5e-3, atol=5e-3` of eager;
- greedy token and next position are identical;
- every unaddressed K/V element is bitwise unchanged;
- logs contain actual `te_exactqk` and `te_bf16barrier` `LaunchKernel` records;
- physical NPU 7 is empty before and after execution.

## Decision

- Pass: freeze G4a evidence and begin G4b Device UDF epoch work.
- Finite but inaccurate: localize the earliest differing layer/operator.
- Non-finite output: reject the precision-policy hypothesis and instrument
  intermediate tensors before another full native run.
- Model-copy failure: precision mode is coupled to native1's failure; isolate
  the remaining GE settings rather than entering G4b.

## Boundary

This attempt can pass G4a only. Device-side greedy, EOS, recurrence and the
single Feed/Fetch epoch remain G4b.
