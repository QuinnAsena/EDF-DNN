"""Tests for edf_dnn.data.normalise, on a tiny pixel table.

Run with:  pixi run test
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from edf_dnn.data.normalise import load_scaler, save_scaler

VARIABLES = ["elevation", "forest"]

# Elevations in metres and forest fractions: very different ranges
TRAINING = pd.DataFrame({
    "cell_id": [1, 2, 3, 4],
    "elevation": np.array([100, 300, 500, 1100], dtype="float32"),
    "forest": np.array([0.0, 0.2, 0.5, 1.0], dtype="float32"),
})


def test_saved_scaler_scales_exactly_like_the_original(tmp_path):
    scaler = MinMaxScaler().fit(TRAINING[VARIABLES])

    save_scaler(scaler, tmp_path)
    loaded = load_scaler(tmp_path)

    # Same numbers, so predictions on the future grid use exactly the
    # scaling the model was trained with
    assert np.array_equal(loaded.transform(TRAINING[VARIABLES]),
                          scaler.transform(TRAINING[VARIABLES]))
    # and it still knows which variable is which
    assert list(loaded.feature_names_in_) == VARIABLES


def test_scaler_refuses_columns_in_a_different_order():
    # Not our code, but sklearn behaviour we rely on: fitted on a DataFrame,
    # the scaler checks the variable names, so a mix-up fails loudly
    scaler = MinMaxScaler().fit(TRAINING[VARIABLES])

    with pytest.raises(ValueError, match="feature names"):
        scaler.transform(TRAINING[["forest", "elevation"]])
