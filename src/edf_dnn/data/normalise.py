"""Scaling predictors using statistics from the training folds only.

Why this module exists
----------------------
Neural networks train poorly when inputs have very different ranges
(e.g. elevation in metres vs. forest fraction 0-1). The scaling statistics
must come from training data only; computing them on all data leaks
information about the validation/test areas into training.

Planned contents (phase 3)
--------------------------
- Compute per-variable statistics (min/max or mean/sd) over training cells.
- Apply them to any stack or batch.
- Save them inside each run folder, so predictions on the future grid use
  exactly the same scaling the model was trained with.
"""
