---
status: superseded by ADR-0008
---

# Keep bounded Host-visible Decode epochs on Device

A Host-Visible Decode Epoch uses one Device graph invocation, with the NPU
owning the Decode loop, sampling, EOS and active state, and KV progression for
up to K steps. The sidecar owns lifecycle, epoch admission, result transport,
and fault containment, but the Host starts every next epoch and therefore
retains steady-state scheduling control. This mechanism proved control-plane
amortization and remains useful as a measurement scaffold, but it no longer
qualifies as the target Decode Control Offload architecture.
