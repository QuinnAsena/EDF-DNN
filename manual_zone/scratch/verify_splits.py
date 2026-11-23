"""Interactive walkthrough of edf_dnn.data.splits (hold_out_test, assign_folds, downsample) on the real data.

Run it line by line with Shift+Enter (console starts in the repo root), or all
at once with:  pixi run python manual_zone/scratch/verify_splits.py
(as a whole script the maps are skipped unless you add plt.show()).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from edf_dnn.data.grid import crop_to_grid, open_stack, to_pixel_table
from edf_dnn.data.splits import assign_folds, downsample, hold_out_test

# ---------------------------------------------------------------------------
# 1. The pixel table, as in verify_grid.py, and the split settings
# ---------------------------------------------------------------------------
config = yaml.safe_load(open("configs/data.yaml")) # loads up configs
target, seed = config["target"], config["random_seed"] # target = burned01 column (burned/unburned)
print(config["test_fraction"], config["n_folds"], seed)       # 0.1 5 1984

stack = open_stack(f"{config['data_dir']}/{config['historical_raster']}",
                   [target] + config["variables"])
template = rioxarray.open_rasterio(f"{config['data_dir']}/{config['future_raster']}")
historical_data = to_pixel_table(crop_to_grid(stack, template))
print(len(historical_data), round(historical_data[target].mean(), 3))  # 563,475 pixels, 0.137 burned

# ---------------------------------------------------------------------------
# 2. Held-out test set: step through hold_out_test
# ---------------------------------------------------------------------------
# `pool` = the portion of the observed data not in test
# the _pool_ is used in cross-validation by splitting into folds
# `test` = the held-out test set, not used in cross-validation
pool, test = train_test_split(historical_data, test_size=config["test_fraction"],
                              stratify=historical_data[target], random_state=seed)
# train_test_split creates pool and test with ~equal burned vs unburned pixels
# No balancing/downsampling has been done at this stage and both pool and test
# are majority unburned
print(len(pool), len(test))                    # 507,127 and 56,348: a 90/10 split
print(round(pool[target].mean(), 4), round(test[target].mean(), 4))   # same burned share: stratified
print(test.index[:5])                          # the rows come back shuffled...
pool, test = pool.sort_index(), test.sort_index()   # ...so put them back in map order
print(test.index[:5])

# Now the real function, and independent checks
pool_function, test_function = hold_out_test(historical_data, target, config["test_fraction"], seed)
print(pool_function.equals(pool), test_function.equals(test))   # True True
print(set(pool["cell_id"]).isdisjoint(test["cell_id"]))         # True: no pixel in both
print(len(pool) + len(test) == len(historical_data))            # True: none lost

# ---------------------------------------------------------------------------
# 3. Cross-validation folds: step through assign_folds
# ---------------------------------------------------------------------------
# Splits the pool into folds
splitter = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=seed)
for fold, (training_positions, validation_positions) in enumerate(
        splitter.split(pool, pool[target]), start=1):
    print(fold, len(training_positions), len(validation_positions))   # ~405,700 train, ~101,400 validate

# split() hands out row POSITIONS (0, 1, 2... within the pool), not row labels.
# The pool's labels have gaps where the test rows were, so the two differ:
print(validation_positions[:5])                # positions in the pool
print(pool.index[validation_positions[:5]])    # the matching row labels

# Now the real function: one fold number per pool pixel
folds = assign_folds(pool, target, config["n_folds"], seed)
print(folds.value_counts().sort_index())                   # ~101,400 pixels per fold
print(pool.groupby(folds)[target].mean().round(4))         # 0.137 in every fold: stratified

# One fold's training and validation pixels, as run.py will take them:
# fold 1 validates, the other four train
pool["fold"] = folds
training, validation = pool[pool["fold"] != 1], pool[pool["fold"] == 1]
print(len(training), len(validation))                      # ~405,700 and ~101,400
print(set(training["cell_id"]).isdisjoint(validation["cell_id"]))   # True
# Over the 5 folds, every pool pixel validates exactly once
print(pool["fold"].between(1, 5).all())                    # True

# ---------------------------------------------------------------------------
# 4. Maps: where the test and fold pixels are
# ---------------------------------------------------------------------------
# Put each pixel's split back on the grid by its cell_id, like
# terra's r[cells] <- values: cell_id - 1 is its position in the flattened grid
split_code = np.full(template.rio.height * template.rio.width, np.nan)
split_code[test["cell_id"] - 1] = 0                        # 0 = test
split_code[pool["cell_id"] - 1] = pool["fold"]             # 1-5 = fold
split_map = xr.DataArray(split_code.reshape(template.rio.height, template.rio.width),
                         dims=("y", "x"), coords={"y": template.y, "x": template.x})

# Six classes, one colour each: levels are the bin edges between the codes
map_style = {"levels": np.arange(-0.5, 6), "cmap": "tab10",
             "cbar_kwargs": {"ticks": range(6), "label": "0 = test, 1-5 = fold"}}
# **map_style passes the dictionary's entries as named arguments, like do.call() in R
split_map.plot(**map_style)                                # from afar: one colourful smear
# Zoom in on a 40 x 40 km window in the middle of the data: neighbouring
# pixels sit in different folds and in the test set ("salt and pepper").
# A model can learn a pixel's neighbours in training and be checked on the
# pixel itself, which flatters random-fold scores: the reason for spatial
# block folds later (Roberts et al. 2017)
middle_row = int(np.median((pool["cell_id"] - 1) // template.rio.width))
middle_column = int(np.median((pool["cell_id"] - 1) % template.rio.width))
split_map.isel(y=slice(middle_row - 20, middle_row + 20),
               x=slice(middle_column - 20, middle_column + 20)).plot(**map_style)

# ---------------------------------------------------------------------------
# 5. Downsampling one fold's training pixels: step through downsample
# ---------------------------------------------------------------------------
burned = training[training[target] == 1]
unburned = training[training[target] == 0].sample(n=len(burned), random_state=seed)
print(len(burned), len(unburned))                          # equal: 55,433 each
# The fold is downsampled to 50/50 burned/unburned
balanced = downsample(training, target, seed)
print(len(training), len(balanced))                        # 405,701 -> 110,866 pixels
print(balanced[target].value_counts())                     # 50/50
print(balanced[target].sum() == training[target].sum())    # True: every burned pixel kept
print(set(balanced["cell_id"]) <= set(training["cell_id"]))   # True: only training pixels
# Validation pixels are NOT downsampled: they keep the real burned share
print(round(validation[target].mean(), 3))                 # 0.137

# ---------------------------------------------------------------------------
# 6. Same seed, same splits: compare with a run folder, if one exists
# ---------------------------------------------------------------------------
splits_file = Path("runs/dev_quick/splits.csv")            # written by run.py, step (a)
if splits_file.exists():
    saved = pd.read_csv(splits_file).set_index("cell_id")["split"]
    ours = pd.concat([pd.Series("test", index=test["cell_id"]),
                      "fold_" + pool.set_index("cell_id")["fold"].astype(str)])
    print((saved == ours.loc[saved.index]).all())          # True: identical assignment

# Tidy up before exit, which avoids rioxarray's harmless shutdown message
del stack, template
import gc
gc.collect()
