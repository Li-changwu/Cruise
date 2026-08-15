---
status: accepted
---

# Bound the first model path

P3 through P5 use one pinned Qwen2.5-7B-Instruct revision and tokenizer at TP=1
and PP=1 with a single primary EOS and fully specified greedy sampling. The
Persistent path rejects unsupported temperature, top-k, top-p, penalty, stop,
and speculative-decoding semantics instead of falling back to Host-controlled
Decode. The Decode-Heavy Target alone sets `ignore_eos` to guarantee 256 measured
output tokens; separate cases preserve and verify normal EOS behavior. Broader
model and sampling compatibility follows proof of the core architecture and
cannot block or dilute its qualification.
