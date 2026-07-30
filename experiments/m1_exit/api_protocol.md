# M1 OpenAI API Differential Protocol

The gate starts an actual `vllm serve` process for the stock and Cruise routes
in sequence, with `--no-async-scheduling` so both use the synchronous EngineCore
contract required by the resident scheduler. Both expose `/v1/completions` with
the same model, tokenizer, request bodies, and NPU. Responses enable vLLM's `return_token_ids` extension
so comparison uses exact token IDs rather than decoded text alone.

The cases cover single and batched non-streaming responses, single and batched
SSE streams, `max_tokens` finish reasons, EOS on a deterministic second output
token, a deterministic unsupported `min_tokens=1` request, and a stream closed
after three tokens. A final request must succeed after the disconnect.
Per-choice token order, finish and stop reasons, prompt indices, usage
accounting, and the `[DONE]` boundary must match.

Each server is placed in its own process group and receives SIGTERM after the
client gate. Any remaining group is killed after a bounded grace period. The
storage guard verifies NPU, process, root-disk, and `/dev/shm` cleanup between
the stock and Cruise routes and again at finalization.

The versioned manifest names the qualified tokenizer path used on the primary
test host. A second host may set `CRUISE_API_TOKENIZER` to the same frozen model
revision without editing the manifest; model config and weight identities remain
checked by the hardware driver.
