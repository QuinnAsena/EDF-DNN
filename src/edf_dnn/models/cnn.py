"""Patch CNN: predicts burn probability for the centre pixel of a patch.

Planned (phase 4): a few convolution + pooling blocks over a
(variables x height x width) patch, then a small fully connected head
producing one logit, the same output convention as the MLPs.
"""
