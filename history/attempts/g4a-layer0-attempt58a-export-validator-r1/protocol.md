# Attempt 58a Export Validator r1

Date frozen: 2026-07-24

The original Attempt 58a export completed and produced all model artifacts,
but its added graph inspector returned 91 because it read only concrete GE
`shape.dim` fields. The private converter records these shapes in `_meta`.

This validator does not rerun or overwrite the export. It reads the immutable
Attempt 58a AIR, pbtxt, eager reference, ABI and original status; accepts the
known original status only when export/eager/ABI succeeded and the sole failure
is graph inspection 91; then validates the gate operands from `_meta` and
rechecks all node counts. The result is evidence about the verifier only, not a
new model experiment.
