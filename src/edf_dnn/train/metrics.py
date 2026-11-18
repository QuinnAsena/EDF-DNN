"""All evaluation metrics, computed in one place.

Contents
--------
- compute_metrics : true labels + predicted probabilities -> burned share,
                    ROC AUC, average precision, Brier score and skill score,
                    log loss, and recall, specificity, precision and TSS at a
                    threshold.

Every model, every fold and the final test set is scored by this one function,
so the numbers are always comparable. (geocomp-2025's train.py repeated these
lines in three places.)
"""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def compute_metrics(labels, probabilities, threshold=0.5):
    """Score predicted burn probabilities against what actually burned.

    Parameters
    ----------
    labels        : the true 0/1 values (burned01), one per pixel. Must contain
                    both burned and unburned pixels.
    probabilities : predicted burn probabilities for the same pixels, in the same
                    order. Probabilities, not logits: apply torch.sigmoid first.
    threshold     : a pixel counts as "predicted burned" above this probability.
                    Only recall, specificity, precision and TSS use it; the
                    other metrics look at the probabilities themselves.

    Returns
    -------
    A dict of plain floats: burned_share, roc_auc, average_precision, brier,
    brier_skill_score, log_loss, recall, specificity, precision,
    true_skill_statistic.
    """
    # Plain 1D numpy arrays of floats; ravel flattens e.g. a (n, 1) model output to (n,)
    labels = np.asarray(labels, dtype=float).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()

    # Fail loudly on input that would give quiet nonsense. With only one class
    # (e.g. a spatial fold without any burns), sklearn returns NaN for ROC AUC
    # with just a warning. Logits instead of probabilities would leave ROC AUC
    # unchanged (same ranking) but make Brier and recall meaningless.
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError(f"labels must contain both 0 and 1, got values {np.unique(labels)}")
    if probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("probabilities must lie between 0 and 1: were logits passed "
                         "without torch.sigmoid?")

    # Share of these pixels that burned. The baseline for average precision
    # and the Brier skill score; with spatial folds it differs between folds.
    burned_share = labels.mean()

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
    # 0.7 should burn about 70% of the time).
    # Brier (1950), Monthly Weather Review 78: 1-3, the original definition:
    # https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
    brier = np.mean((probabilities - labels) ** 2)

    # Brier skill score: the Brier score relative to a no-skill reference that
    # always predicts the burned share, which scores burned_share * (1 - burned_share)
    # (about 0.118 at 13.7% burned). 1 = perfect, 0 = no better than the
    # reference, below 0 = worse. Comparable across folds with different burn rates.
    # Wheatcroft (2019), Int. J. Forecasting 35: 573-579, on reading skill scores:
    # https://doi.org/10.1016/j.ijforecast.2018.11.010
    brier_skill_score = 1 - brier / (burned_share * (1 - burned_share))

    # Log loss: the binary cross-entropy, i.e. the training loss without
    # pos_weight. Like Brier, but punishes confident mistakes much harder
    # (a burned pixel given 0.01 costs far more than one given 0.3). Lower is better.
    log_loss_value = log_loss(labels, probabilities)

    # The confusion matrix counts at one threshold (like table(observed, predicted) in R)
    predicted_burned = probabilities > threshold
    actually_burned = labels == 1
    true_positives = np.sum(predicted_burned & actually_burned)     # burned, and caught
    false_positives = np.sum(predicted_burned & ~actually_burned)   # false alarm
    false_negatives = np.sum(~predicted_burned & actually_burned)   # burned, but missed
    true_negatives = np.sum(~predicted_burned & ~actually_burned)   # unburned, correctly

    # Recall (sensitivity): of the pixels that burned, the share the model caught.
    # The ecologically important number. (Never 0 / 0: both classes are present.)
    recall = true_positives / (true_positives + false_negatives)

    # Specificity: of the pixels that did not burn, the share correctly left out
    specificity = true_negatives / (true_negatives + false_positives)

    # Precision: of the pixels predicted burned, the share that did burn.
    # 0 rather than an error if no pixel is predicted burned.
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    # True skill statistic (TSS, Youden's J): recall + specificity - 1.
    # 0 = no better than random, 1 = perfect. Unlike accuracy or kappa it does
    # not depend on the burned share, which is why it is standard in species
    # distribution modelling. Allouche et al. (2006), J. Applied Ecology 43:
    # 1223-1232, https://doi.org/10.1111/j.1365-2664.2006.01214.x
    true_skill_statistic = recall + specificity - 1

    # float() turns numpy numbers into plain Python numbers, which save cleanly
    # to JSON/CSV in the run folders
    return {
        "burned_share": float(burned_share),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "brier": float(brier),
        "brier_skill_score": float(brier_skill_score),
        "log_loss": float(log_loss_value),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "true_skill_statistic": float(true_skill_statistic),
    }
