"""All evaluation metrics, computed in one place.

Contents
--------
- compute_metrics : true labels + predicted probabilities -> ROC AUC, average
                    precision, Brier score, recall and precision.

Every model, every fold and the final test set is scored by this one function,
so the numbers are always comparable. (geocomp-2025's train.py repeated these
lines in three places.)
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(labels, probabilities, threshold=0.5):
    """Score predicted burn probabilities against what actually burned.

    Parameters
    ----------
    labels        : the true 0/1 values (burned01), one per pixel
    probabilities : predicted burn probabilities for the same pixels, in the same
                    order. Probabilities, not logits: apply torch.sigmoid first.
    threshold     : a pixel counts as "predicted burned" above this probability.
                    Only recall and precision use it; the other three metrics
                    look at the probabilities themselves.

    Returns
    -------
    A dict of plain floats: roc_auc, average_precision, brier, recall, precision.
    """
    # Plain 1D numpy arrays of floats; ravel flattens e.g. a (n, 1) model output to (n,)
    labels = np.asarray(labels, dtype=float).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()

    # ROC AUC: the chance that a random burned pixel gets a higher probability
    # than a random unburned one. 0.5 = coin toss, 1 = perfect. Uses only the
    # ranking of the probabilities, across all thresholds. Like pROC::auc() in R.
    roc_auc = roc_auc_score(labels, probabilities)

    # Average precision: area under the precision-recall curve. It looks only at
    # how well the burned class is found, so it is more telling than ROC AUC
    # when burns are rare. A useless model scores the burned share (about 0.14).
    # The old code used the trapezoid rule on the PR curve; but the curve is a
    # staircase, and straight lines between its corners misstate the area.
    # Average precision adds up the steps instead (sklearn's recommended summary).
    average_precision = average_precision_score(labels, probabilities)

    # Brier score: mean squared difference between probability and outcome,
    # BS = mean((p - y)^2). 0 is perfect; lower is better. It rewards
    # probabilities that are both well ranked and well calibrated (a predicted
    # 0.7 should burn about 70% of the time). Baseline: always predicting the
    # burned share (0.137) scores 0.137 * (1 - 0.137) = about 0.118.
    # Wheatcroft (2019), Int. J. Forecasting 35: 573-579, on Brier skill scores:
    # https://doi.org/10.1016/j.ijforecast.2018.11.010
    brier = np.mean((probabilities - labels) ** 2)

    # Recall and precision at one threshold, from the confusion matrix counts
    # (like table(observed, predicted) in R)
    predicted_burned = probabilities > threshold
    actually_burned = labels == 1
    true_positives = np.sum(predicted_burned & actually_burned)     # burned, and caught
    false_positives = np.sum(predicted_burned & ~actually_burned)   # false alarm
    false_negatives = np.sum(~predicted_burned & actually_burned)   # burned, but missed

    # Recall (sensitivity): of the pixels that burned, the share the model caught.
    # The ecologically important number. 0 rather than an error if nothing burned.
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    # Precision: of the pixels predicted burned, the share that did burn.
    # 0 rather than an error if no pixel is predicted burned.
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    # float() turns numpy numbers into plain Python numbers, which save cleanly
    # to JSON/CSV in the run folders
    return {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "brier": float(brier),
        "recall": float(recall),
        "precision": float(precision),
    }
