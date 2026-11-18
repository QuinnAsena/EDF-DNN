"""Tests for edf_dnn.train.metrics, on tiny examples small enough to work out by hand.

Run with:  pixi run test
"""

import numpy as np
import pytest

from edf_dnn.train.metrics import compute_metrics

# Four pixels: two unburned, two burned. Sorted by probability, the order is
# 0.9 (burned), 0.6 (unburned), 0.4 (burned), 0.1 (unburned): good, not perfect.
LABELS = [0, 0, 1, 1]
PROBABILITIES = [0.1, 0.6, 0.4, 0.9]


def test_metrics_match_hand_calculation():
    metrics = compute_metrics(LABELS, PROBABILITIES)

    # ROC AUC: of the 2 x 2 (burned, unburned) pairs, the burned pixel scores
    # higher in 3: 0.4 > 0.1, 0.9 > 0.1, 0.9 > 0.6 (but not 0.4 > 0.6)
    assert metrics["roc_auc"] == pytest.approx(3 / 4)
    # Average precision: walk down the sorted list; each time a burned pixel
    # is found, add (its share of the burned pixels) x (precision so far):
    # 0.9 -> 1/2 x 1/1, then 0.4 -> 1/2 x 2/3
    assert metrics["average_precision"] == pytest.approx(1 / 2 * 1 + 1 / 2 * 2 / 3)
    # Brier: squared errors 0.1², 0.6², 0.6², 0.1², averaged
    assert metrics["brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)
    # At 0.5, pixels 0.6 and 0.9 are predicted burned: 1 caught (0.9),
    # 1 false alarm (0.6), 1 missed (0.4)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["precision"] == pytest.approx(1 / 2)


def test_threshold_moves_recall_and_precision_only():
    metrics = compute_metrics(LABELS, PROBABILITIES, threshold=0.3)

    # At 0.3, the 0.4 pixel is now caught too: both burned pixels found
    # (recall 1), with 1 false alarm among 3 predicted burned (precision 2/3)
    assert metrics["recall"] == pytest.approx(1)
    assert metrics["precision"] == pytest.approx(2 / 3)
    # The threshold-free metrics don't change
    default = compute_metrics(LABELS, PROBABILITIES)
    for name in ["roc_auc", "average_precision", "brier"]:
        assert metrics[name] == default[name]


def test_perfect_predictions():
    metrics = compute_metrics(LABELS, [0.0, 0.0, 1.0, 1.0])

    assert metrics == {"roc_auc": 1.0, "average_precision": 1.0, "brier": 0.0,
                       "recall": 1.0, "precision": 1.0}


def test_nothing_predicted_burned_gives_zero_not_an_error():
    # Every probability is below the threshold, so precision would be 0 / 0
    metrics = compute_metrics(LABELS, [0.1, 0.2, 0.3, 0.4])

    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0


def test_column_shaped_model_output_is_accepted():
    # A model returns one column per pixel, shape (4, 1); labels may be (4,)
    column = np.array(PROBABILITIES).reshape(4, 1)

    assert compute_metrics(LABELS, column) == compute_metrics(LABELS, PROBABILITIES)
