# M1 Runner Diagnostics on NPU0-1

These short notes retain the causes of failed attempts without retaining the
large remote logs or scratch trees.

| Run | Result | Cause and correction |
|---|---|---|
| EngineCore r1 | stopped before NPU execution | A path-contract test assumed pytest `tmp_path` was outside `/dev/shm`; the test was made location-independent. |
| EngineCore r2 | baseline passed; Cruise stopped at the second cohort | The generated DataFlow deploy root was the old `/root/runtime/deploy_res`; it now uses marker-owned `/dev/shm` scratch. |
| EngineCore r3 | GraphPp load OOM | Stock vLLM and Cruise now share an explicit 512 MiB KV-cache budget instead of the unconstrained profiled cache. |
| EngineCore r4 | passed | 1,000-request differential passed on physical NPU 0. |
| API r1 | stopped before model execution | The case manifest retained an old tokenizer path; the runner now defaults the tokenizer to the selected frozen model directory. |
| API r2 | both servers passed; compare stage failed | Shared runner omitted `--cases` for API comparison; the argument is now passed and covered by a contract test. |
| API r3 | passed | All eight API semantics cases and the final differential comparison passed. |

Failed run directories were removed after these diagnoses were recorded.
