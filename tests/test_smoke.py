"""Smoke test: the package installs, imports, and a model runs a forward pass.

If this fails, the environment is broken, not the science.
Run with:  pixi run test
"""

import torch

from edf_dnn.models.mlp import FeedforwardTwo


def test_mlp_forward_pass_shape():
    number_of_pixels = 8
    number_of_variables = 25
    model = FeedforwardTwo(input_size=number_of_variables, hidden_size=16, dropout=0.1)
    logits = model(torch.randn(number_of_pixels, number_of_variables))
    # One raw logit per pixel
    assert logits.shape == (number_of_pixels, 1)
