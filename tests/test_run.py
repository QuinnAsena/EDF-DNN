"""Tests for edf_dnn.train.run, step (a): run folder, configs, pixel table, splits.

All writing goes to pytest's temporary folders, never to the real runs/.
Run with:  pixi run test
"""

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

import edf_dnn.models.mlp
from edf_dnn.train.run import (
    REPO_ROOT,
    load_pixel_table,
    prepare_run_folder,
    save_splits,
)

DATA = {"target": "burned01", "variables": ["elevation"]}


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
    # name would overwrite each other's run folder), and its architecture
    # must be a model class that exists
    paths = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
    assert paths
    for path in paths:
        experiment = yaml.safe_load(open(path))
        assert experiment["name"] == path.stem, path.name
        assert hasattr(edf_dnn.models.mlp, experiment["model"]["architecture"]), path.name


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

    table = load_pixel_table(data)

    assert list(table.columns) == ["cell_id", "x", "y", "burned01", "elevation"]
    assert table["cell_id"].tolist() == [1, 2, 3, 4]               # cell numbers on the future grid
    assert table["elevation"].tolist() == [5.0, 6.0, 9.0, 10.0]   # historical cells 5, 6, 9, 10


def test_splits_file_lists_every_pixel_once(tmp_path):
    pool = pd.DataFrame({"cell_id": [1, 2, 4, 5], "fold": [1, 2, 1, 2]})
    test = pd.DataFrame({"cell_id": [3]})

    save_splits(pool, test, tmp_path)

    splits = pd.read_csv(tmp_path / "splits.csv")
    assert splits["cell_id"].tolist() == [1, 2, 3, 4, 5]
    assert splits["split"].tolist() == ["fold_1", "fold_2", "test", "fold_1", "fold_2"]
