"""Tests for edf_dnn.models: every architecture builds the same way, returns one logit per pixel, and is found by name.

Run with:  pixi run test
"""

import pytest
import torch

from edf_dnn.models import get_architecture
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


def test_architectures_are_found_by_name():
    # The configs say e.g. `architecture: FeedforwardTwo`; run.py looks the class up by that name
    assert get_architecture("FeedforwardSingle") is FeedforwardSingle
    assert get_architecture("FeedforwardTwo") is FeedforwardTwo
    assert get_architecture("FeedforwardTapered") is FeedforwardTapered
    # A typo gives an error that lists the valid names
    with pytest.raises(ValueError, match="FeedforwardTwo"):
        get_architecture("FeedForwardTwo")
