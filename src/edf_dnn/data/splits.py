"""Which pixels are used for what: held-out test set, cross-validation folds, downsampling.

Contents
--------
- hold_out_test : split off a stratified test set, scored once at the very end.
- assign_folds  : give every remaining pixel a fold number (random, stratified).
- downsample    : all burned pixels plus as many randomly chosen unburned ones.

Random stratified folds, as in geocomp-2025, so the old AUCs are a fair
check. Spatial block folds (blocks.py) will later replace assign_folds; the
rest of the pipeline only needs a fold number per pixel either way.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def hold_out_test(table, target, test_fraction, seed):
    """Split pixels into a training pool and a held-out test set.

    Stratified: both keep the table's burned share. The test set is never
    seen during cross-validation or model selection, only scored at the end.

    Returns
    -------
    (pool, test): two pixel tables, each in the original (map) row order
    """
    # Like rsample::initial_split(table, prop = 0.9, strata = burned01) in R
    pool, test = train_test_split(table, test_size=test_fraction,
                                  stratify=table[target], random_state=seed)
    # train_test_split shuffles the rows; put them back in map order
    return pool.sort_index(), test.sort_index()


def assign_folds(table, target, n_folds, seed):
    """Give every pixel a fold number from 1 to n_folds.

    Stratified: every fold keeps the table's burned share. Like
    rsample::vfold_cv(table, v = 5, strata = burned01) in R.

    Returns
    -------
    A Series of fold numbers, one per row of the table (same index), named "fold"
    """
    folds = pd.Series(0, index=table.index, name="fold")
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # split() yields, for each fold in turn, the row POSITIONS of its training
    # and validation pixels; we only need the validation ones to label each fold
    for fold, (_, validation_positions) in enumerate(splitter.split(table, table[target]), start=1):
        folds.iloc[validation_positions] = fold   # iloc: by position, not by row label
    return folds


def downsample(table, target, seed):
    """All burned pixels plus the same number of randomly chosen unburned pixels (50/50).

    Applied to training pixels only: validation and test pixels keep the real
    burned share, as the map will.
    """
    burned = table[table[target] == 1]
    # sample(n=...) draws without replacement, like dplyr::slice_sample(n = ...)
    unburned = table[table[target] == 0].sample(n=len(burned), random_state=seed)
    return pd.concat([burned, unburned]).sort_index()   # back in map order
