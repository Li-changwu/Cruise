---
status: accepted
---

# Fail stop on owner loss

The first Persistent architecture does not reconstruct Device KV state after an
Owner Instance fails. It preserves the last acknowledged Committed Prefix,
fails every request still owned by that instance without emitting additional
tokens, and quarantines its Device KV Leases until reset or equivalent
invalidation makes reuse safe. A replacement receives a new owner identity so
late events and acknowledgements cannot cross the failure boundary. This gives
up transparent continuation to prevent duplicated output and split Device state
while the Persistent control path is being established.
