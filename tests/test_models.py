"""Tests for edf_dnn.models.mlp: every architecture builds the same way and returns one logit per pixel.

Run with:  pixi run test
"""

import pytest
import torch

from edf_dnn.models.mlp import FeedforwardSingle, FeedforwardTapered, FeedforwardTwo


@pytest.mark.parametrize("architecture", [FeedforwardSingle, FeedforwardTwo, FeedforwardTapered])
def test_one_logit_per_pixel(architecture):
    # The shared constructor run.py relies on: (input_size, hidden_size, dropout)
    model = architecture(input_size=25, hidden_size=16, dropout=0.1)
    batch = torch.rand(7, 25)   # 7 pixels, 25 predictors

    logits = model(batch)

    assert logits.shape == (7, 1)


def test_tapered_second_layer_is_half_as_wide():
    model = FeedforwardTapered(input_size=25, hidden_size=16, dropout=0.1)

    assert model.first_hidden_layer.out_features == 16
    assert model.second_hidden_layer.out_features == 8
