"""Entry point: train one experiment described by a config file.

Usage (from the repo root):
    pixi run python -m edf_dnn.train.run configs/experiments/mlp_two_layer.yaml

Reads the experiment config plus configs/data.yaml, and writes everything to
runs/<experiment name>/. If that folder already exists, the run stops unless
the config sets `overwrite: true`.

Built in three steps (see docs/TODO.md):
  (a) configs, run folder, pixel table, test set and folds   <- done
  (b) one fold, then the grid search over hyperparameters
  (c) final retrain on the whole pool, scored once on the test set
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import rioxarray
import yaml

from edf_dnn.data.grid import crop_to_grid, open_stack, to_pixel_table
from edf_dnn.data.splits import assign_folds, hold_out_test

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


def main(experiment_path):
    experiment, data = load_configs(experiment_path)
    target, seed = data["target"], data["random_seed"]
    run_dir = prepare_run_folder(experiment, data)

    table = load_pixel_table(data)
    pool, test = hold_out_test(table, target, data["test_fraction"], seed)
    pool["fold"] = assign_folds(pool, target, data["n_folds"], seed)
    save_splits(pool, test, run_dir)

    print(f"Run folder: {run_dir}")
    print(f"Pixels: {len(table):,}; test {len(test):,} (burned {test[target].mean():.3f}), "
          f"cross-validation pool {len(pool):,} (burned {pool[target].mean():.3f})")
    print(pool.groupby("fold")[target].agg(pixels="size", burned_share="mean").round(3))
    # Steps (b) and (c) follow here: grid search over the folds, then the final model


if __name__ == "__main__":
    # Runs only when started as a command (python -m edf_dnn.train.run ...),
    # not when imported, e.g. by the tests
    parser = argparse.ArgumentParser(description="Train one experiment from its config file.")
    parser.add_argument("config", help="experiment YAML, e.g. configs/experiments/mlp_two_layer.yaml")
    main(parser.parse_args().config)
