"""Entry point: train one experiment described by a config file.

Usage (from the repo root):
    pixi run python -m edf_dnn.train.run configs/experiments/mlp_two_layer.yaml

Reads the experiment config plus configs/data.yaml, and writes everything to
runs/<experiment name>/. If that folder already exists, the run stops unless
the config sets `overwrite: true`.

Built in three steps (see docs/TODO.md):
  (a) configs, run folder, pixel table, test set and folds   <- done
  (b) every hyperparameter setting on every fold             <- done
  (c) final retrain on the whole pool, scored once on the test set

Files in runs/<name>/ so far:
  config.yaml, git_commit.txt  what produced the run
  splits.csv                   test pixels, and each other pixel's fold
  cv_folds.csv                 one row per setting and fold: best epoch, minutes, validation metrics
  cv_results.csv               one row per setting: mean and sd over its folds, best setting first
  cv_history.csv               every epoch of every setting and fold (learning curves)
  cv_predictions.parquet       every validation pixel's probability, per setting and fold
The cv_ files are rewritten after every fold, so a crash loses only the fold in progress.
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import rioxarray
import torch
import yaml
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler

from edf_dnn.data.batching import batch_loader, pixel_dataset
from edf_dnn.data.grid import crop_to_grid, open_stack, to_pixel_table
from edf_dnn.data.splits import assign_folds, downsample, hold_out_test
from edf_dnn.models import get_architecture
from edf_dnn.train.loop import choose_device, fit, predict_probabilities
from edf_dnn.train.metrics import compute_metrics

# The repo's top folder, found from this file's location
# (src/edf_dnn/train/run.py -> up 3 folders), so relative paths in the configs
# work whatever folder the command is started from (e.g. a cluster job)
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_configs(experiment_path):
    """Read an experiment config and the shared data config, as plain dictionaries."""
    with open(experiment_path) as file:
        experiment = yaml.safe_load(file)   # YAML -> nested dicts and lists, like yaml::read_yaml()
    with open(REPO_ROOT / "configs" / "data.yaml") as file:
        data = yaml.safe_load(file)
    return experiment, data


def git_commit():
    """The current git commit, and whether there were uncommitted changes."""
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD") or "unknown"
    changed = "yes" if git("status", "--porcelain") else "no"
    return f"commit: {commit}\nuncommitted changes: {changed}\n"


def prepare_run_folder(experiment, data, runs_dir=REPO_ROOT / "runs"):
    """Create runs/<name>/ and record what produced the run in it.

    Writes config.yaml (both configs, as used) and git_commit.txt. An
    existing folder is only replaced when the experiment says `overwrite: true`.
    """
    name = experiment["name"]
    # The folder gets deleted when overwriting, so the name must be one plain
    # folder name: an empty name or ".." would point at runs/ itself or above
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"Experiment name must be a plain folder name, got {name!r}")

    run_dir = Path(runs_dir) / name
    if run_dir.exists():
        if not experiment.get("overwrite", False):
            raise FileExistsError(f"{run_dir} already exists; set overwrite: true to replace it")
        shutil.rmtree(run_dir)   # delete the old run, like unlink(run_dir, recursive = TRUE)
    run_dir.mkdir(parents=True)

    # Record exactly what produced this run
    with open(run_dir / "config.yaml", "w") as file:
        yaml.safe_dump({"experiment": experiment, "data": data}, file, sort_keys=False)
    (run_dir / "git_commit.txt").write_text(git_commit())
    return run_dir


def load_pixel_table(data):
    """Historical stack -> cropped to the future grid -> one row per complete pixel."""
    # An absolute data_dir (e.g. /glade/... on the cluster) is used as it is;
    # a relative one is taken from the repo root
    data_dir = REPO_ROOT / data["data_dir"]
    stack = open_stack(data_dir / data["historical_raster"], [data["target"]] + data["variables"])
    template = rioxarray.open_rasterio(data_dir / data["future_raster"])
    return to_pixel_table(crop_to_grid(stack, template))


def save_splits(pool, test, run_dir):
    """Write splits.csv: which pixels are test pixels, and each other pixel's fold."""
    splits = pd.concat([
        pd.DataFrame({"cell_id": test["cell_id"], "split": "test"}),
        pd.DataFrame({"cell_id": pool["cell_id"], "split": "fold_" + pool["fold"].astype(str)}),
    ]).sort_values("cell_id")
    splits.to_csv(run_dir / "splits.csv", index=False)


def grid_settings(experiment):
    """Every combination of the listed hyperparameter values, like expand.grid() in R.

    Returns a list of dicts, e.g. {"dropout": 0.1, "hidden_size": 128, "learning_rate": 0.001}.
    """
    return list(ParameterGrid({
        "hidden_size": experiment["model"]["hidden_size"],
        "dropout": experiment["model"]["dropout"],
        "learning_rate": experiment["training"]["learning_rate"],
    }))


def train_one_fold(pool, fold, setting, experiment, data, device):
    """Train a fresh model on every fold but `fold`, then predict the pixels of `fold`.

    The same steps as manual_zone/scratch/walkthrough_mlp.py, sections 3 to 7.

    Returns
    -------
    history     : one row per epoch (training loss, validation metrics)
    best_epoch  : the epoch whose weights the model kept
    predictions : cell_id, label, probability for every validation pixel
    """
    target, variables, seed = data["target"], data["variables"], data["random_seed"]
    training_settings = experiment["training"]
    training, validation = pool[pool["fold"] != fold], pool[pool["fold"] == fold]

    # Scaler from this fold's training pixels only, all of them (before downsampling)
    scaler = MinMaxScaler().fit(training[variables])
    balanced = downsample(training, target, seed)
    batch_size = training_settings["batch_size"]
    training_loader = batch_loader(pixel_dataset(balanced, variables, scaler, target),
                                   batch_size, shuffle=True)
    validation_loader = batch_loader(pixel_dataset(validation, variables, scaler, target),
                                     batch_size, shuffle=False)

    # Reset the random numbers before every model, like set.seed() before each
    # model in R: starting weights, batch order and dropout are then the same
    # whatever order the settings and folds run in, so any one can be rerun alone
    torch.manual_seed(seed)
    architecture = get_architecture(experiment["model"]["architecture"])
    model = architecture(len(variables), setting["hidden_size"], setting["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=setting["learning_rate"])
    loss_function = torch.nn.BCEWithLogitsLoss()

    history, best_epoch = fit(model, training_loader, validation_loader, loss_function,
                              optimizer, device, training_settings["max_epochs"],
                              training_settings["patience"])
    predictions = predict_probabilities(model, validation_loader, device)
    return history, best_epoch, predictions


def summarise_settings(fold_results, hyperparameters):
    """One row per setting: mean and sd of each validation metric over its folds, and its best epochs."""
    folds = pd.DataFrame(fold_results)
    metrics = [name for name in folds.columns if name.startswith("validation_")]
    by_setting = folds.groupby(["setting", *hyperparameters])   # like dplyr::group_by()
    summary = pd.concat([
        by_setting.size().rename("folds_done"),
        by_setting["best_epoch"].median().rename("best_epoch_median"),
        by_setting["best_epoch"].mean().rename("best_epoch_mean"),
        by_setting[metrics].mean().add_suffix("_mean"),
        by_setting[metrics].std().add_suffix("_sd"),   # pandas std = R's sd() (n - 1)
    ], axis=1)
    return summary.reset_index().sort_values("validation_roc_auc_mean", ascending=False)


def save_cross_validation(histories, fold_results, predictions, hyperparameters, run_dir):
    """Write the cv_ files from everything finished so far; returns the per-setting summary."""
    summary = summarise_settings(fold_results, hyperparameters)
    pd.DataFrame(fold_results).to_csv(run_dir / "cv_folds.csv", index=False)
    summary.to_csv(run_dir / "cv_results.csv", index=False)
    pd.concat(histories).to_csv(run_dir / "cv_history.csv", index=False)
    # Millions of rows: parquet is smaller and faster than CSV, and keeps the
    # column types (in R: arrow::read_parquet())
    pd.concat(predictions).to_parquet(run_dir / "cv_predictions.parquet", index=False)
    return summary


def cross_validate(pool, experiment, data, device, run_dir):
    """Train every hyperparameter setting on every fold, saving the results after each fold.

    Returns the per-setting summary (as in cv_results.csv), best mean validation ROC AUC first.
    """
    settings = grid_settings(experiment)
    hyperparameters = list(settings[0])   # the setting's names: dropout, hidden_size, learning_rate
    histories, fold_results, predictions = [], [], []

    for setting_number, setting in enumerate(settings, start=1):
        for fold in range(1, data["n_folds"] + 1):
            started = time.perf_counter()
            history, best_epoch, fold_predictions = train_one_fold(pool, fold, setting,
                                                                   experiment, data, device)
            minutes = (time.perf_counter() - started) / 60
            metrics = compute_metrics(fold_predictions["label"], fold_predictions["probability"])

            # Label every row with its setting and fold, so the runs can be told apart
            labels = {"setting": setting_number, **setting, "fold": fold}
            histories.append(history.assign(**labels)[[*labels, *history.columns]])
            predictions.append(fold_predictions.assign(setting=setting_number, fold=fold)
                               [["setting", "fold", "cell_id", "label", "probability"]])
            fold_results.append({**labels, "best_epoch": best_epoch, "epochs_run": len(history),
                                 "minutes": round(minutes, 2),
                                 **{f"validation_{name}": value for name, value in metrics.items()}})

            # Rewrite the files now, so a crash (or a laptop restart) mid-grid
            # loses only the fold in progress
            summary = save_cross_validation(histories, fold_results, predictions,
                                            hyperparameters, run_dir)
            # flush=True: show the line straight away, also when output goes to a log file
            print(f"setting {setting_number}/{len(settings)} {setting}, fold {fold}: "
                  f"best epoch {best_epoch} of {len(history)}, "
                  f"validation ROC AUC {metrics['roc_auc']:.3f}, {minutes:.1f} min", flush=True)
    return summary


def main(experiment_path):
    experiment, data = load_configs(experiment_path)
    target, seed = data["target"], data["random_seed"]
    run_dir = prepare_run_folder(experiment, data)

    historical_data = load_pixel_table(data)
    pool, test = hold_out_test(historical_data, target, data["test_fraction"], seed)
    pool["fold"] = assign_folds(pool, target, data["n_folds"], seed)
    save_splits(pool, test, run_dir)

    print(f"Run folder: {run_dir}")
    print(f"Pixels: {len(historical_data):,}; test {len(test):,} (burned {test[target].mean():.3f}), "
          f"cross-validation pool {len(pool):,} (burned {pool[target].mean():.3f})")
    print(pool.groupby("fold")[target].agg(pixels="size", burned_share="mean").round(3))

    # Step (b): every hyperparameter setting on every fold
    device = choose_device(experiment["device"])
    print(f"Device: {device}", flush=True)
    summary = cross_validate(pool, experiment, data, device, run_dir)
    # The headline columns only; cv_results.csv has every metric
    print(summary[["setting", "hidden_size", "dropout", "learning_rate", "best_epoch_median",
                   "validation_roc_auc_mean", "validation_roc_auc_sd",
                   "validation_average_precision_mean"]].round(4).to_string(index=False))
    # Step (c) follows here: retrain the best setting on the whole pool, score once on test


if __name__ == "__main__":
    # Runs only when started as a command (python -m edf_dnn.train.run ...),
    # not when imported, e.g. by the tests
    parser = argparse.ArgumentParser(description="Train one experiment from its config file.")
    parser.add_argument("config", help="experiment YAML, e.g. configs/experiments/mlp_two_layer.yaml")
    main(parser.parse_args().config)
