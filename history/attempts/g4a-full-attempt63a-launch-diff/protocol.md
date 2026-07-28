# Attempt 63a launch-plan differential

Compare the immutable Attempt 62a and Attempt 63a native GE logs without using
an NPU. The purpose is to determine whether exposing 16 layer-0 graph outputs
only adds output transfers or changes compiled operator fusion/kernel choices.

This is a post-hoc diagnostic over preserved logs. It cannot validate Attempt
63a, pass G4a, or justify entering G4b.
