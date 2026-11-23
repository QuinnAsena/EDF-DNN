"""Tests for edf_dnn.train.run: run folder, configs, pixel table, splits (step a),
training on the folds and saving the results (step b).

All writing goes to pytest's temporary folders, never to the real runs/.
Run with:  pixi run test
"""

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_origin
from sklearn.preprocessing import MinMaxScaler

import edf_dnn.train.run as run
from edf_dnn.data.splits import assign_folds
from edf_dnn.models import ARCHITECTURES
from edf_dnn.train.run import (
    REPO_ROOT,
    cross_validate,
    grid_settings,
    load_pixel_table,
    prepare_run_folder,
    save_splits,
    train_one_fold,
)

DATA = {"target": "burned01", "variables": ["elevation"]}
CPU = torch.device("cpu")


def test_run_folder_records_what_produced_the_run(tmp_path):
    experiment = {"name": "test_run", "overwrite": False, "model": {"hidden_size": [8]}}

    run_dir = prepare_run_folder(experiment, DATA, runs_dir=tmp_path)

    assert run_dir == tmp_path / "test_run"
    saved = yaml.safe_load(open(run_dir / "config.yaml"))
    assert saved == {"experiment": experiment, "data": DATA}
    assert (run_dir / "git_commit.txt").read_text().startswith("commit:")


def test_existing_run_folder_is_only_replaced_with_overwrite(tmp_path):
    experiment = {"name": "test_run", "overwrite": False}
    run_dir = prepare_run_folder(experiment, DATA, runs_dir=tmp_path)
    (run_dir / "old_result.csv").write_text("from the previous run")

    # Without overwrite: refuse, and leave the old run alone
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_run_folder(experiment, DATA, runs_dir=tmp_path)
    assert (run_dir / "old_result.csv").exists()

    # With overwrite: a fresh, empty run folder
    prepare_run_folder({**experiment, "overwrite": True}, DATA, runs_dir=tmp_path)
    assert not (run_dir / "old_result.csv").exists()


@pytest.mark.parametrize("bad_name", ["", ".", "..", "nested/name"])
def test_names_that_could_delete_other_folders_are_refused(tmp_path, bad_name):
    (tmp_path / "other_run").mkdir()

    with pytest.raises(ValueError, match="plain folder name"):
        prepare_run_folder({"name": bad_name, "overwrite": True}, DATA, runs_dir=tmp_path)
    assert (tmp_path / "other_run").exists()


def test_experiment_configs_are_consistent():
    # Each config's name must match its file name (two configs with the same
    # name would overwrite each other's run folder), its architecture must be
    # a model class that exists, and its grid must build
    paths = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
    assert paths
    for path in paths:
        experiment = yaml.safe_load(open(path))
        assert experiment["name"] == path.stem, path.name
        assert experiment["model"]["architecture"] in ARCHITECTURES, path.name
        assert grid_settings(experiment), path.name


def write_raster(path, band_values, band_names, left, top):
    """A small float32 GeoTIFF with named bands, 1 km cells, top-left corner at (left, top)."""
    band_count, height, width = band_values.shape
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                       count=band_count, dtype="float32", crs="EPSG:3857",
                       transform=from_origin(left, top, 1000, 1000), nodata=-9999.0) as raster:
        for band_number, name in enumerate(band_names, start=1):
            raster.write(band_values[band_number - 1].astype("float32"), band_number)
            raster.set_band_description(band_number, name)


def test_pixel_table_is_cropped_to_the_future_grid(tmp_path):
    # Historical: 3 x 4 cells; future grid: the 2 x 2 block one row down and
    # one column in. Elevation = the historical cell number, to see which cells came
    elevation = np.arange(12, dtype="float32").reshape(3, 4)
    write_raster(tmp_path / "historical.tif", np.stack([np.ones((3, 4)), elevation]),
                 ["burned01", "elevation"], left=0, top=3000)
    write_raster(tmp_path / "future.tif", np.zeros((1, 2, 2)), ["anything"], left=1000, top=2000)
    data = {**DATA, "data_dir": str(tmp_path),   # an absolute path is used as it is
            "historical_raster": "historical.tif", "future_raster": "future.tif"}

    historical_data = load_pixel_table(data)

    assert list(historical_data.columns) == ["cell_id", "x", "y", "burned01", "elevation"]
    assert historical_data["cell_id"].tolist() == [1, 2, 3, 4]              # cell numbers on the future grid
    assert historical_data["elevation"].tolist() == [5.0, 6.0, 9.0, 10.0]  # historical cells 5, 6, 9, 10


def test_splits_file_lists_every_pixel_once(tmp_path):
    pool = pd.DataFrame({"cell_id": [1, 2, 4, 5], "fold": [1, 2, 1, 2]})
    test = pd.DataFrame({"cell_id": [3]})

    save_splits(pool, test, tmp_path)

    splits = pd.read_csv(tmp_path / "splits.csv")
    assert splits["cell_id"].tolist() == [1, 2, 3, 4, 5]
    assert splits["split"].tolist() == ["fold_1", "fold_2", "test", "fold_1", "fold_2"]


# ---------------------------------------------------------------------------
# Step (b): every setting on every fold, on 300 made-up pixels
# ---------------------------------------------------------------------------
FOLD_DATA = {"target": "burned01", "variables": ["first", "second"],
             "random_seed": 1984, "n_folds": 3}
EXPERIMENT = {
    "model": {"architecture": "FeedforwardSingle", "hidden_size": [8], "dropout": [0.0]},
    "training": {"learning_rate": [0.01], "batch_size": 64, "max_epochs": 5, "patience": 3},
}


def make_pool(pixel_count=300):
    """Pixels with two random predictors; 'burned' when the first is above 0.6 (about 40%)."""
    generator = np.random.default_rng(1984)
    pool = pd.DataFrame({"cell_id": np.arange(1, pixel_count + 1),
                         "first": generator.random(pixel_count),
                         "second": generator.random(pixel_count)})
    pool["burned01"] = (pool["first"] > 0.6).astype(float)
    pool["fold"] = assign_folds(pool, "burned01", FOLD_DATA["n_folds"], seed=1984)
    return pool


def test_grid_has_every_combination():
    experiment = {"model": {"hidden_size": [128, 256], "dropout": [0.1, 0.2]},
                  "training": {"learning_rate": [1e-3, 5e-4, 1e-4]}}

    settings = grid_settings(experiment)

    assert len(settings) == 2 * 2 * 3
    assert {"hidden_size": 256, "dropout": 0.1, "learning_rate": 1e-4} in settings
    assert len({tuple(sorted(setting.items())) for setting in settings}) == 12   # no repeats


def test_one_fold_predicts_exactly_its_validation_pixels():
    pool = make_pool()
    setting = grid_settings(EXPERIMENT)[0]

    history, best_epoch, predictions = train_one_fold(pool, 2, setting, EXPERIMENT, FOLD_DATA, CPU)

    assert sorted(predictions["cell_id"]) == sorted(pool.loc[pool["fold"] == 2, "cell_id"])
    assert predictions["probability"].between(0, 1).all()
    assert 1 <= len(history) <= 5                   # never more than max_epochs
    assert 1 <= best_epoch <= len(history)


def test_scaler_is_fitted_on_all_training_pixels_and_nothing_else(monkeypatch):
    # Not on validation pixels (leakage), and not on the downsampled pixels only
    pool = make_pool()
    fitted_on = []

    class RecordingScaler(MinMaxScaler):
        """MinMaxScaler that notes which rows it was fitted on."""
        def fit(self, features, *arguments, **keywords):
            fitted_on.append(features.index)
            return super().fit(features, *arguments, **keywords)

    monkeypatch.setattr(run, "MinMaxScaler", RecordingScaler)   # swap in the recording version
    train_one_fold(pool, 2, grid_settings(EXPERIMENT)[0], EXPERIMENT, FOLD_DATA, CPU)

    assert fitted_on[0].equals(pool.index[pool["fold"] != 2])


def test_one_fold_gives_the_same_result_when_rerun():
    # The seed is reset inside train_one_fold, so earlier random numbers don't matter
    pool = make_pool()
    setting = grid_settings(EXPERIMENT)[0]

    _, _, first_run = train_one_fold(pool, 1, setting, EXPERIMENT, FOLD_DATA, CPU)
    torch.rand(100)                                  # use up some random numbers in between
    _, _, second_run = train_one_fold(pool, 1, setting, EXPERIMENT, FOLD_DATA, CPU)

    assert first_run.equals(second_run)


def test_cross_validation_files_cover_every_setting_and_fold(tmp_path):
    pool = make_pool()
    experiment = {**EXPERIMENT, "model": {**EXPERIMENT["model"], "hidden_size": [4, 8]}}   # 2 settings

    summary = cross_validate(pool, experiment, FOLD_DATA, CPU, tmp_path)

    folds = pd.read_csv(tmp_path / "cv_folds.csv")
    assert len(folds) == 2 * 3                                   # 2 settings x 3 folds
    results = pd.read_csv(tmp_path / "cv_results.csv")
    assert results["folds_done"].tolist() == [3, 3]
    assert results["validation_roc_auc_mean"].is_monotonic_decreasing   # best setting first
    assert summary["setting"].tolist() == results["setting"].tolist()
    history = pd.read_csv(tmp_path / "cv_history.csv")
    assert set(zip(history["setting"], history["fold"])) == set(zip(folds["setting"], folds["fold"]))
    # Every pool pixel is predicted exactly once per setting (by the model that didn't train on it)
    predictions = pd.read_parquet(tmp_path / "cv_predictions.parquet")
    for _, one_setting in predictions.groupby("setting"):
        assert sorted(one_setting["cell_id"]) == pool["cell_id"].tolist()


def test_results_are_saved_after_every_fold(tmp_path, monkeypatch):
    # Simulate a crash in the second fold: the first fold must already be on disk
    real_train_one_fold = run.train_one_fold
    calls = []

    def crash_on_second_fold(*arguments):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        return real_train_one_fold(*arguments)

    monkeypatch.setattr(run, "train_one_fold", crash_on_second_fold)   # swap in the crashing version
    with pytest.raises(RuntimeError, match="simulated crash"):
        cross_validate(make_pool(), EXPERIMENT, FOLD_DATA, CPU, tmp_path)

    assert len(pd.read_csv(tmp_path / "cv_folds.csv")) == 1
