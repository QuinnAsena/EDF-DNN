"""Tests for edf_dnn.train.metrics, on tiny examples small enough to work out by hand.

Run with:  pixi run test
"""

import math

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score

from edf_dnn.train.metrics import compute_metrics

# Four pixels: two unburned, two burned. Sorted by probability, the order is
# 0.9 (burned), 0.6 (unburned), 0.4 (burned), 0.1 (unburned): good, not perfect.
LABELS = [0, 0, 1, 1]
PROBABILITIES = [0.1, 0.6, 0.4, 0.9]


def test_metrics_match_hand_calculation():
    metrics = compute_metrics(LABELS, PROBABILITIES)

    assert metrics["burned_share"] == pytest.approx(2 / 4)
    # ROC AUC: of the 2 x 2 (burned, unburned) pairs, the burned pixel scores
    # higher in 3: 0.4 > 0.1, 0.9 > 0.1, 0.9 > 0.6 (but not 0.4 > 0.6)
    assert metrics["roc_auc"] == pytest.approx(3 / 4)
    # Average precision: walk down the sorted list; each time a burned pixel
    # is found, add (its share of the burned pixels) x (precision so far):
    # 0.9 -> 1/2 x 1/1, then 0.4 -> 1/2 x 2/3
    assert metrics["average_precision"] == pytest.approx(1 / 2 * 1 + 1 / 2 * 2 / 3)
    # Brier: squared errors 0.1², 0.6², 0.6², 0.1², averaged
    assert metrics["brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)
    # Brier skill score: the reference (always predict 0.5) scores 0.5 x 0.5
    assert metrics["brier_skill_score"] == pytest.approx(1 - 0.185 / 0.25)
    # Log loss: minus the log of the probability given to what actually
    # happened (0.9, 0.4, 0.4, 0.9), averaged
    expected_log_loss = -(math.log(0.9) + math.log(0.4) + math.log(0.4) + math.log(0.9)) / 4
    assert metrics["log_loss"] == pytest.approx(expected_log_loss)
    # At 0.5, pixels 0.6 and 0.9 are predicted burned: 1 caught (0.9),
    # 1 false alarm (0.6), 1 missed (0.4), 1 correctly left out (0.1)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["specificity"] == pytest.approx(1 / 2)
    assert metrics["precision"] == pytest.approx(1 / 2)
    # TSS = 1/2 + 1/2 - 1: at this threshold, no better than random
    assert metrics["true_skill_statistic"] == pytest.approx(0)


def test_threshold_moves_only_the_threshold_metrics():
    metrics = compute_metrics(LABELS, PROBABILITIES, threshold=0.3)

    # At 0.3, the 0.4 pixel is now caught too: both burned pixels found
    # (recall 1), with 1 false alarm among 3 predicted burned (precision 2/3)
    assert metrics["recall"] == pytest.approx(1)
    assert metrics["specificity"] == pytest.approx(1 / 2)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["true_skill_statistic"] == pytest.approx(1 / 2)
    # The threshold-free metrics don't change
    default = compute_metrics(LABELS, PROBABILITIES)
    for name in ["burned_share", "roc_auc", "average_precision", "brier",
                 "brier_skill_score", "log_loss"]:
        assert metrics[name] == default[name]


def test_perfect_predictions():
    metrics = compute_metrics(LABELS, [0.0, 0.0, 1.0, 1.0])

    # approx: log loss is a tiny number, not exactly 0 (sklearn clips
    # probabilities just inside 0 and 1 so the log stays finite)
    assert metrics == pytest.approx({
        "burned_share": 0.5, "roc_auc": 1.0, "average_precision": 1.0,
        "brier": 0.0, "brier_skill_score": 1.0, "log_loss": 0.0,
        "recall": 1.0, "specificity": 1.0, "precision": 1.0,
        "true_skill_statistic": 1.0,
    }, abs=1e-9)


def test_predicting_the_burned_share_has_no_skill():
    # 1 of 4 pixels burned. Predicting 0.25 everywhere is the no-skill
    # reference itself, so its Brier skill score must be exactly 0.
    metrics = compute_metrics([0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25])

    # Brier = 0.25 x 0.75: three errors of 0.25² and one of 0.75², averaged
    assert metrics["brier"] == pytest.approx(0.25 * 0.75)
    assert metrics["brier_skill_score"] == pytest.approx(0)
    # No ranking at all: a coin toss for ROC AUC, the burned share for AP
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(0.25)


def test_nothing_predicted_burned_gives_zero_not_an_error():
    # Every probability is below the threshold, so precision would be 0 / 0
    metrics = compute_metrics(LABELS, [0.1, 0.2, 0.3, 0.4])

    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["specificity"] == 1.0


def test_column_shaped_model_output_is_accepted():
    # A model returns one column per pixel, shape (4, 1); labels may be (4,)
    column = np.array(PROBABILITIES).reshape(4, 1)

    assert compute_metrics(LABELS, column) == compute_metrics(LABELS, PROBABILITIES)


def test_only_one_class_raises():
    # e.g. a validation fold without a single burned pixel
    with pytest.raises(ValueError, match="both 0 and 1"):
        compute_metrics([0, 0, 0, 0], PROBABILITIES)


def test_logits_instead_of_probabilities_raise():
    logits = [-2.2, 0.4, -0.4, 2.2]   # raw model output, before the sigmoid

    with pytest.raises(ValueError, match="sigmoid"):
        compute_metrics(LABELS, logits)


def test_agrees_with_sklearn_on_random_data():
    # Independent check on a bigger example: sklearn's own versions of the
    # metrics we compute by hand
    generator = np.random.default_rng(1984)
    labels = generator.integers(0, 2, size=200)
    probabilities = generator.random(200)
    predicted_burned = probabilities > 0.5

    metrics = compute_metrics(labels, probabilities)

    assert metrics["brier"] == pytest.approx(brier_score_loss(labels, probabilities))
    # Specificity is the recall of the unburned class
    assert metrics["specificity"] == pytest.approx(
        recall_score(labels, predicted_burned, pos_label=0))
    # sklearn's "adjusted" balanced accuracy is the TSS (Youden's J)
    assert metrics["true_skill_statistic"] == pytest.approx(
        balanced_accuracy_score(labels, predicted_burned, adjusted=True))
