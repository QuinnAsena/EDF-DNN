"""Tests for edf_dnn.data.splits, on 1000 made-up pixels.

Run with:  pixi run test
"""

import pandas as pd
import pytest

from edf_dnn.data.splits import assign_folds, downsample, hold_out_test

# 1000 pixels; every 7th one burned (142 burned, a 14.2% burned share)
TABLE = pd.DataFrame({"cell_id": range(1, 1001)})
TABLE["burned01"] = (TABLE["cell_id"] % 7 == 0).astype(float)
BURNED_SHARE = TABLE["burned01"].mean()


def test_test_set_is_stratified_and_separate():
    pool, test = hold_out_test(TABLE, "burned01", test_fraction=0.1, seed=1984)

    assert (len(pool), len(test)) == (900, 100)
    # No pixel in both, none lost
    assert set(pool["cell_id"]).isdisjoint(test["cell_id"])
    assert len(set(pool["cell_id"]) | set(test["cell_id"])) == 1000
    # Both keep the burned share (stratified), within one pixel's worth
    assert test["burned01"].mean() == pytest.approx(BURNED_SHARE, abs=0.01)
    assert pool["burned01"].mean() == pytest.approx(BURNED_SHARE, abs=0.01)
    # Back in map order
    assert pool["cell_id"].is_monotonic_increasing and test["cell_id"].is_monotonic_increasing


def test_every_pixel_gets_one_fold():
    folds = assign_folds(TABLE, "burned01", n_folds=5, seed=1984)

    assert folds.index.equals(TABLE.index)
    assert sorted(folds.unique()) == [1, 2, 3, 4, 5]
    assert folds.value_counts().tolist() == [200] * 5
    # Every fold keeps the burned share (stratified)
    for fold in range(1, 6):
        assert TABLE.loc[folds == fold, "burned01"].mean() == pytest.approx(BURNED_SHARE, abs=0.01)


def test_folds_follow_the_seed():
    first = assign_folds(TABLE, "burned01", n_folds=5, seed=1984)

    assert first.equals(assign_folds(TABLE, "burned01", n_folds=5, seed=1984))
    assert not first.equals(assign_folds(TABLE, "burned01", n_folds=5, seed=1985))


def test_folds_work_on_a_pool_with_gaps_in_its_row_labels():
    # After hold_out_test, the pool's row labels have gaps (the test rows are
    # gone); folds are assigned by position, so every pool pixel still gets one
    pool, _ = hold_out_test(TABLE, "burned01", test_fraction=0.1, seed=1984)

    folds = assign_folds(pool, "burned01", n_folds=5, seed=1984)

    assert folds.index.equals(pool.index)
    assert (folds > 0).all()


def test_downsample_balances_and_keeps_every_burned_pixel():
    balanced = downsample(TABLE, "burned01", seed=1984)

    burned = balanced[balanced["burned01"] == 1]
    unburned = balanced[balanced["burned01"] == 0]
    assert len(burned) == len(unburned) == 142
    # All burned pixels kept; the unburned ones are real pixels from the table
    assert set(burned["cell_id"]) == set(TABLE.loc[TABLE["burned01"] == 1, "cell_id"])
    assert set(unburned["cell_id"]) <= set(TABLE["cell_id"])
    # Same seed, same sample
    assert balanced.equals(downsample(TABLE, "burned01", seed=1984))
