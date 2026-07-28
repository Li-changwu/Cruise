# Device-Resident Decode Control: Research Status (2026-07-22)

## Executive Verdict

The central mechanism is feasible on the available Ascend 910B2 server.
DataFlow Device UDFs can keep bounded iteration control on the device, invoke a
real Qwen layer-0 attention/KV AIR repeatedly, maintain recurrent state, select
between registered graph routes, stop on EOS or a step bound, and return through
one Host Feed/Fetch transaction.

Current gate status:

| Gate | Status | Evidence |
|---|---|---|
| Platform and Device UDF availability | PASS | 8 x 910B2, CANN 9.0.0, DataFlow toolchain and aarch64 UDF compilation |
| Synthetic recurrent control | PASS | N Host submissions reduced to 1; exact final state |
| Real Qwen layer-slice recurrence | PASS | Host and Device-UDF AIR routes elementwise exact for N=1,2,4 |
| Bounded decode control contract | PASS | EOS/max-step stop, graph selection, recurrent K/V/position, one Feed/Fetch |
| Full eager semantic fidelity | BLOCKED | Sparse QK differences remain in the GE online-compiled path |
| Full decoder and vLLM integration | OPEN | Paged KV, logits/sampling, dynamic batching, and scheduler integration are not implemented |

The correct research claim is therefore: device-resident iteration control is
mechanically and performance-feasible on 910B2 for a controlled real-model
slice. A full LLM decode loop has not yet been demonstrated.

## Server Facts

- SSH target: local research-server alias (redacted from the public source archive).
- Product: eight Ascend 910B2 devices.
- CANN: 9.0.0; `npu-smi`: 26.0.rc1.
- All experiments in this status used physical NPU 7 after confirming it had
  no device process. Other occupied devices were not touched.
- DataFlow Python, C++ headers, `meta_flow_func.h`, device libraries, and the
  aarch64 cross compiler are installed and usable.
- The available system profiling path exported TS CPU PMU/top-function data.
  It did not expose inner AI Core task timestamps, Ctrl CPU CSVs, or AI CPU
  CSVs, so no AI Core inter-iteration gap reduction is claimed.

## Experimental Evidence

### Synthetic P0

Verdict: `P0_FEASIBILITY_PASS_FULL_QWEN_ROUTE_BLOCKED`.

The Device UDF uses one Host Feed/Fetch independent of N, while Host GE uses N
`RunGraph` calls. All final states were exact. The crossover was N=2.

| N | Host GE (us) | Device UDF (us) | Speedup | Host CPU reduction |
|---:|---:|---:|---:|---:|
| 1 | 837.0 | 921.0 | 0.91x | -30.2% |
| 2 | 1663.5 | 1136.0 | 1.46x | 32.6% |
| 4 | 3324.0 | 1551.5 | 2.14x | 66.2% |
| 32 | 24951.0 | 6842.5 | 3.65x | 94.6% |

### Real-Qwen Attention/KV P0

Verdict: `REAL_QWEN_P0_MECHANISM_PASS_AIR_EAGER_SEMANTICS_BLOCKED`.

All 30 measured Host/Device comparisons were elementwise exact for attention,
K cache, V cache, and position. This compares recurrence routes for the same
AIR; it does not make that AIR eager-equivalent.

| N | Host Feed/Fetch | Device Feed/Fetch | Speedup | Host CPU reduction |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0.86x | -0.1% |
| 2 | 2 | 1 | 1.47x | 49.9% |
| 4 | 4 | 1 | 3.95x | 72.8% |

### Bounded Decode Controller

Verdict: `BOUNDED_DECODE_CONTROL_PASS_FULL_DECODER_BLOCKED`.

The implemented transaction is:

```text
Host Feed(hidden, position, K, V, control)
  -> Device UDF
     -> choose decode_graph_0 or decode_graph_1
     -> RunFlowModel
     -> update position/K/V and control counters
     -> derive synthetic token
     -> stop on EOS or max_steps; otherwise repeat
  -> Host Fetch(final attention, K, V, position, control)
```

The runtime control input is:

```text
[max_steps, eos_token, eos_after_step, graph_switch_step,
 token_seed, token_stride]
```

The returned control state appends executed steps, final token, finish reason,
per-route invocation counts, and final position. The UDF checks staged-hidden
and K/V capacity before the first model invocation.

| Scenario | Steps | Stop | Host Feed/Fetch | Device Feed/Fetch | Speedup | Host CPU reduction | Exact |
|---|---:|---|---:|---:|---:|---:|:---:|
| max1 | 1 | max | 1 | 1 | 0.91x | -18.7% | yes |
| eos2_of4 | 2 | EOS | 2 | 1 | 1.66x | 41.4% | yes |
| eos3_of4 | 3 | EOS | 3 | 1 | 2.59x | 65.5% | yes |
| max4_switch2 | 4 | max | 4 | 1 | 3.19x | 72.7% | yes |

Across the 40 measured comparisons, attention, K cache, V cache, position, and
the complete control tensor were elementwise exact. Both device graph keys were
exercised. The N=1 negative regime is consistent with fixed DataFlow/UDF
transaction overhead; benefits begin when two or more Host round trips are
removed.

### Capacity Boundary

The original test matrix included an eight-call case. It exposed a contract
violation: the AIR has eight KV slots but only four staged hidden rows, so the
fifth invocation indexes beyond the valid hidden table. In preserved
`raw-run2`, the control tensor remained exact but model tensors became
non-deterministic. This is negative boundary evidence, not a valid performance
sample.

The controller now rejects such a transaction before `RunFlowModel`. A separate
probe requested eight steps at initial position 0 and observed hidden rows=4,
KV slots=8, nonzero return code, and zero outputs. The expected rejection test
passed.

## QK Semantic Boundary (Attempts 41-49)

- Attempt 41 built the fixed-shape static-tiling AIC QK package.
- Attempts 42-43 executed it through native GE, then found sparse mismatches:
  one of 224 values at position 1 and four of 224 at position 3, maximum error
  0.25 (about one BF16 ULP in the affected values).
- Attempt 44 ran the same operands by direct custom-kernel launch on physical
  NPU 7 three times per position. All 12 launches were deterministic and
  elementwise exact versus native NPU BMM and eager.
- Attempts 45-46 changed only GE schedule mode from 1 to 0. Native execution
  succeeded and produced the same sparse mismatch.
- Attempts 47-49 passed the exact 72 tiling bytes as a third ordinary input.
  Native GE executed with `arg_size=48`; outputs were byte-identical to the
  static mode-0/mode-1 results.

Facts exclude card variation, schedule mode, and tiling-address/source as the
sole cause. The remaining difference is associated with the GE online-compiled
execution/code-generation path. This attribution does not prove the precise
compiler transformation responsible, and exact eager semantics must not be
claimed for the current full-Qwen route.

## Claim Boundary

Directly proved:

- Iteration-level control can execute inside a Device UDF on 910B2.
- Host submission count can change from N to one.
- Device-side recurrence can preserve the same AIR's model state exactly over
  the valid four-step domain.
- Device-side EOS/max-step decisions and runtime graph-key selection work.
- The controlled benchmarks show a repeatable crossover at two iterations and
  increasing Host CPU savings as the bounded epoch grows.

Supported system inference:

- A decode runtime can profit from executing short, bounded device-resident
  epochs instead of returning to Host after every token.
- The likely integration unit is not an unbounded autonomous scheduler. It is
  a bounded epoch with explicit capacity, stop, and fallback contracts.

Not proved or implemented:

- Full Qwen decoder correctness or eager equivalence.
- Real logits, sampling, or model-generated EOS.
- vLLM paged-KV/block-table updates.
- Per-request active masks, slot compaction, continuous batching, request
  admission, preemption, or cancellation.
- Distinct batch/shape-specialized graph binaries behind the two route keys.
- Multi-card tensor/pipeline parallel coordination.
- Inner AI Core idle-gap reduction, because current profiling cannot observe
  the enclosed model task timestamps.

## Next Gate

The next implementation should replace the synthetic token contract with one
known-correct fixed-batch decoder step that returns logits and paged-KV state.
The Device UDF should then add greedy sampling, real EOS, an active-slot mask,
and a bounded epoch length. Only after that passes Host/Device and eager
correctness should the vLLM-Ascend scheduler hand off a batch epoch to the
device controller.

A minimal integration contract is:

```text
Host -> device:
  fixed batch slots, block tables, sequence lengths, max epoch steps,
  sampling configuration, graph-variant table, cancellation generation

Device-owned epoch:
  select graph -> decoder step -> sample -> update paged KV/length/mask
  -> stop each slot on EOS/limit -> stop epoch on empty batch/bound

Device -> Host:
  generated token spans, completion reasons, final lengths/KV metadata,
  unfinished-slot continuation state, error/fallback code
```

The Host remains responsible for admission, global fairness, memory allocation,
cross-request scheduling, and recovery. The device owns only the latency-critical
bounded inner loop.

## Artifacts

- Synthetic P0: `results/p0-final/`
- Real-Qwen P0: `results/real-qwen-p0/`
- QK Attempts 41-49: `results/g2g-attempt41-49/`
- Bounded controller results: `results/bounded-decode-controller/`
- Bounded controller source: `work/bounded-decode-controller/`
- Remote bounded-controller root:
  `/root/ascend-control-bounded-decode-20260722/`

The successful raw result SHA-256 is
`cdcc789d030ac340024f24597f620147259b29cf80854376d9d745f318cfa5be`.
The summary SHA-256 is
`3dc4cac07427c4e79902a77a4f6ee5f1897a6ed9c3c52438d3dedac3680576ca`.
