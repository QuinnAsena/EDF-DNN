"""Do a run folder's files agree with each other and with the data? Checks on runs/dev_quick/.

Nothing is retrained here: every check recomputes a saved number from other
saved numbers, or from the pixel table, and compares.

1. The run folder: what produced it, then read its files back in
2. Pixels: each prediction sits in its own fold; no test pixels; every pool pixel once
3. Labels: the saved labels match burned01 in the data, joined by cell_id
4. Metrics: recomputed from the saved predictions, they equal cv_folds.csv
5. Summary: cv_results.csv equals the fold results averaged by setting
6. History: cv_history.csv agrees with cv_folds.csv about each fold's best epoch

Set run_dir to any finished run (e.g. runs/mlp_two_layer) to check it the same way.

Run it line by line with Shift+Enter (console starts in the repo root), or all
at once (about 15 seconds) with:  pixi run python manual_zone/scratch/verify_run.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from edf_dnn.train.metrics import compute_metrics
from edf_dnn.train.run import load_pixel_table

run_dir = Path("runs/dev_quick")

# ---------------------------------------------------------------------------
# 1. The run folder: what produced it, then read its files back in
# ---------------------------------------------------------------------------
saved_config = yaml.safe_load(open(run_dir / "config.yaml"))
experiment, data = saved_config["experiment"], saved_config["data"]
# True while the config files are unchanged since the run
print(experiment == yaml.safe_load(open(f"configs/experiments/{experiment['name']}.yaml")))
print(data == yaml.safe_load(open("configs/data.yaml")))
print((run_dir / "git_commit.txt").read_text())          # the commit, and whether files had changed

# The files run.py wrote, read back in; the sections below check them against each other
splits =pd.read_csv(run_dir / "splits.csv")
folds = pd.read_csv(run_dir / "cv_folds.csv")             # one row per setting and fold
results = pd.read_csv(run_dir / "cv_results.csv")         # one row per setting
history = pd.read_csv(run_dir / "cv_history.csv")         # one row per setting, fold and epoch
predictions = pd.read_parquet(run_dir / "cv_predictions.parquet")   # one row per setting and pool pixel
print(len(splits), len(folds), len(results), len(history), len(predictions))   # dev_quick: 563475 5 1 100 507127
target = data["target"]
metrics = [name for name in folds.columns if name.startswith("validation_")]   # the 10 metric columns

# splits (splits.csv): which group each historical pixel went into, written before any training.
#   It has 563,475 rows with two columns: cell_id and split. 
#   The split is test (56,348 pixels) or fold_1 to fold_5 (about 101,425 each).
# folds (cv_folds.csv): one row per model trained during cross-validation,
#   so one per setting and fold (5 rows here). Each row holds the hyperparameters, 
#   the best epoch, the epochs run, the minutes taken, and the 10 validation metrics at the best epoch.
# results (cv_results.csv): one row per setting (1 row here). It's folds averaged over
#   the 5 folds: the mean and sd of each metric, and the median and mean best epoch.
#   It's sorted best first, and it's the table you pick the winning setting from.
# history (cv_history.csv): the training curves. There's one row per setting, fold
#  and epoch: 5 folds × 20 epochs = 100 rows. Each row has the training loss 
#  and the 10 validation metrics for that epoch.
# predictions (cv_predictions.parquet): one row per setting and pool pixel (507,127 rows).
#  Each row is the burn probability a pool pixel got from the one model that didn't train on it, with cell_id, label, setting and fold.

# ---------------------------------------------------------------------------
# 2. Pixels: each prediction in its own fold, no test pixels, every pool pixel once
# ---------------------------------------------------------------------------
# splits.csv was written before any training; the predictions after it
split_of_pixel = splits.set_index("cell_id")["split"]     # cell_id -> "test" or "fold_k", like a named vector in R
predicted_in = "fold_" + predictions["fold"].astype(str)
print((split_of_pixel.loc[predictions["cell_id"]].to_numpy() == predicted_in.to_numpy()).all())   # True
test_cells = splits.loc[splits["split"] == "test", "cell_id"]
print(predictions["cell_id"].isin(test_cells).any())      # False: the test set stayed untouched
pool_cells = set(splits.loc[splits["split"] != "test", "cell_id"])
for setting, one_setting in predictions.groupby("setting"):
    # every pool pixel predicted exactly once per setting, by the one model that didn't train on it
    print(setting, one_setting["cell_id"].is_unique and set(one_setting["cell_id"]) == pool_cells)   # True

# ---------------------------------------------------------------------------
# 3. Labels: the saved labels match the data, joined by cell_id
# ---------------------------------------------------------------------------
# The labels travelled through tensors, shuffled batches and back; cell_id
# travelled with them. If the two ever got separated, they would disagree here.
historical_data = load_pixel_table(data)                  # the same function and settings the run used
labelled = predictions.merge(historical_data[["cell_id", target]], on="cell_id", how="left")   # like dplyr::left_join()
print(len(labelled) == len(predictions))                  # True: one match per pixel
print((labelled["label"] == labelled[target]).all())      # True
print(labelled.head())                                    # label and burned01 side by side

# ---------------------------------------------------------------------------
# 4. Metrics: recomputed from the saved predictions, they equal cv_folds.csv
# ---------------------------------------------------------------------------
# One compute_metrics per setting and fold, like lapply(split(predictions, ...), ...) in R
recomputed = pd.DataFrame([
    {"setting": setting, "fold": fold,
     **{f"validation_{name}": value
        for name, value in compute_metrics(group["label"], group["probability"]).items()}}
    for (setting, fold), group in predictions.groupby(["setting", "fold"])
])
compared = folds.merge(recomputed, on=["setting", "fold"], suffixes=("_saved", "_recomputed"))
for name in metrics:
    print(name, np.allclose(compared[f"{name}_saved"], compared[f"{name}_recomputed"], rtol=0, atol=1e-9))   # True
# Validation was scored at the real burned share, not the downsampled 50%
print(folds["validation_burned_share"].round(3).tolist())   # 0.137 in every fold

# ---------------------------------------------------------------------------
# 5. Summary: cv_results.csv equals the fold results averaged by setting
# ---------------------------------------------------------------------------
by_setting = folds.groupby("setting")
saved_summary = results.set_index("setting").sort_index()
for name in metrics:
    same_mean = np.allclose(by_setting[name].mean(), saved_summary[f"{name}_mean"])
    same_sd = np.allclose(by_setting[name].std(), saved_summary[f"{name}_sd"], equal_nan=True)
    print(name, same_mean, same_sd)                        # True True
print(np.allclose(by_setting["best_epoch"].median(), saved_summary["best_epoch_median"]))   # True
print(results["validation_roc_auc_mean"].is_monotonic_decreasing)   # True: best setting first

# ---------------------------------------------------------------------------
# 6. History: cv_history.csv agrees with cv_folds.csv
# ---------------------------------------------------------------------------
# Epochs per fold in the history = epochs_run in cv_folds
epochs_in_history = history.groupby(["setting", "fold"]).size().rename("epochs_in_history")
checked = folds.set_index(["setting", "fold"]).join(epochs_in_history)
print((checked["epochs_in_history"] == checked["epochs_run"]).all())   # True
# The best epoch is the one with the highest validation ROC AUC in each fold's history
highest = history.loc[history.groupby(["setting", "fold"])["validation_roc_auc"].idxmax(),
                      ["setting", "fold", "epoch"]]
compared = folds.merge(highest, on=["setting", "fold"])
print((compared["best_epoch"] == compared["epoch"]).all())   # True
# And that epoch's history row holds the same metrics as cv_folds
best_rows = history.merge(folds[["setting", "fold", "best_epoch"]],
                          left_on=["setting", "fold", "epoch"], right_on=["setting", "fold", "best_epoch"])
compared = folds.merge(best_rows, on=["setting", "fold"], suffixes=("_folds", "_history"))
print(all(np.allclose(compared[f"{name}_folds"], compared[f"{name}_history"]) for name in metrics))   # True
