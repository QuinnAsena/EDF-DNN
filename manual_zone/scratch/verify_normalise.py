"""Interactive walkthrough of the scaling (MinMaxScaler + edf_dnn.data.normalise) on the real data.

Run it line by line with Shift+Enter (console starts in the repo root), or all
at once with:  pixi run python manual_zone/scratch/verify_normalise.py
"""

import tempfile

import numpy as np
import pandas as pd
import rioxarray
import yaml
from sklearn.preprocessing import MinMaxScaler

from edf_dnn.data.grid import crop_to_grid, open_stack, to_pixel_table
from edf_dnn.data.normalise import load_scaler, save_scaler
from edf_dnn.data.splits import assign_folds, downsample, hold_out_test

# ---------------------------------------------------------------------------
# 1. The pixel table and one fold's training and validation pixels, as run.py
#    takes them (see verify_splits.py): fold 1 validates, folds 2-5 train
# ---------------------------------------------------------------------------
config = yaml.safe_load(open("configs/data.yaml"))
target, variables, seed = config["target"], config["variables"], config["random_seed"]
stack = open_stack(f"{config['data_dir']}/{config['historical_raster']}", [target] + variables)
template = rioxarray.open_rasterio(f"{config['data_dir']}/{config['future_raster']}")
historical_data = to_pixel_table(crop_to_grid(stack, template))

pool, test = hold_out_test(historical_data, target, config["test_fraction"], seed)
pool["fold"] = assign_folds(pool, target, config["n_folds"], seed)
training, validation = pool[pool["fold"] != 1], pool[pool["fold"] == 1]
print(len(training), len(validation))                    # 405,701 and 101,426

# ---------------------------------------------------------------------------
# 2. Fit on the training pixels only, then look at what it learned
# ---------------------------------------------------------------------------
# Fit to unbalanced training pixels
scaler = MinMaxScaler().fit(training[variables])
print(pd.DataFrame({"min": scaler.data_min_, "range": scaler.data_range_},
                   index=scaler.feature_names_in_))       # one min and one range PER VARIABLE

# ---------------------------------------------------------------------------
# 3. Apply: numpy arrays (pixels x variables), ready for the network
# ---------------------------------------------------------------------------
def ranges(values):
    """Per-variable min and max of a scaled array, as a small table."""
    return pd.DataFrame(values, columns=variables).agg(["min", "max"]).T

training_values = scaler.transform(training[variables])
print(training_values.shape, training_values.dtype)       # (405701, 25) float32
# Careful: numpy's .min() and .max() reduce over ALL values at once, giving
# one number each, not one per variable (that needs .min(axis=0), like
# apply(m, 2, min) in R, or the ranges() historical_data above)
print(training_values.min(), training_values.max())       # 0.0 1.0: the whole array
print(ranges(training_values))                            # per variable: each 0 to 1 (0.9999999 is float32 rounding)

# After downsampling, values stay inside 0-1 but many variables no longer
# reach both ends: their extreme pixels were among the unburned ones dropped.
# (This is why the scaler is fitted BEFORE downsampling: on all training pixels.)
balanced = downsample(training, target, seed)
balanced_ranges = ranges(scaler.transform(balanced[variables]))
print(balanced_ranges[(balanced_ranges["min"] > 0) | (balanced_ranges["max"] < 1)].round(3))   # e.g. TPI 0.03-0.89

# Validation pixels were not part of the fit, so a few land just outside 0-1
validation_values = scaler.transform(validation[variables])
validation_ranges = ranges(validation_values)
print(validation_ranges[(validation_ranges["min"] < 0) | (validation_ranges["max"] > 1)].round(4))

# ---------------------------------------------------------------------------
# 4. A skewed variable: flow accumulation runs from ~1,000 to ~357 million,
#    so after scaling nearly every pixel sits at ~0 (see NOTES 2026-11-18)
# ---------------------------------------------------------------------------
flowacc = training_values[:, variables.index("flowacc_1km")]
print(np.quantile(flowacc, [0.5, 0.9, 0.99]))           # ~0.00001, 0.0005, 0.06

# ---------------------------------------------------------------------------
# 6. Save and load, as run.py and predict.py will
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as run_dir:          # a throwaway folder
    save_scaler(scaler, run_dir)
    loaded = load_scaler(run_dir)
    print(np.array_equal(loaded.transform(validation[variables]), validation_values))   # True

# Tidy up before exit, which avoids rioxarray's harmless shutdown message
del stack, template
import gc
gc.collect()
