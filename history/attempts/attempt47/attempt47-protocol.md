# G2g Attempt 47: Explicit-Tiling-Input ExactQk Package

## Question

Can CANN package the original dynamic AIC QK computation with its 72-byte
tiling record supplied as a third ordinary `uint8` tensor input, while the
invalid automatically generated trailing tiling argument is retained only for
ABI compatibility and never dereferenced?

## Frozen Variables

- Input package: Attempt 45, SHA-256
  `47be1f718dffd57fd8f0328032e7165d860e6f51d20e955ba1f13503cea3762c`.
- Input static kernel SHA-256:
  `3f11927d58bd570f437fe8ebdedd28772f3db716210505e604334ec518cd39dc`.
- New explicit-input dynamic kernel SHA-256:
  `f3bf8191b6b12c56ff63428d712951f43a4b83046dfc1d93c38712493f5c2363`.
- New three-input op definition SHA-256:
  `47f8cda49a10a6ab9e72cf14f8a5c31ccf0ed95eaedf70baf7d072021d315736`.
- New shape-inference source SHA-256:
  `6f386cebc9365bae455d2bd27ba68d81d47644453818280037c94068ded84812`.
- Host tiling remains byte-identical mode 0, SHA-256
  `6e04aff60c4320def8969ad10a808466db2b5fb1c9a99ab5a3c71ad410f5d89f`.
- The original dynamic switch, all computation classes, fixed A/B/C shapes,
  BF16 types, 24 blocks, and 18-field tiling layout remain unchanged.
- Only the source of the same tiling bytes changes from the automatic trailing
  parameter to the third ordinary graph input.

## Decision Rules

Attempt 47 is build/package-only and does not use an NPU. It passes only if
input hashes match, current artifacts are archived, the three staged sources
match their frozen hashes, offline OPC build and install succeed, the package
changes, the installed kernel is byte-identical, the op registers a required
`DT_UINT8` input, and the generated tiling aliases remain present.

A pass authorizes a separate three-input AIR export. It proves no runtime or
numerical property by itself.
