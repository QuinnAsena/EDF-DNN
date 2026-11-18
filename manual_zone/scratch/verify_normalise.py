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

# ---------------------------------------------------------------------------
# 1. The pixel table, as in verify_grid.py
# ---------------------------------------------------------------------------
config = yaml.safe_load(open("configs/data.yaml"))
variables = config["variables"]
stack = open_stack(f"{config['data_dir']}/{config['historical_raster']}",
                   [config["target"]] + variables)
template = rioxarray.open_rasterio(f"{config['data_dir']}/{config['future_raster']}")
table = to_pixel_table(crop_to_grid(stack, template))
print(len(table))                                  # 563,475 pixels

# A stand-in for one fold: a random 80% of pixels for training, 20% for validation
# (run.py will do this properly, stratified)
generator = np.random.default_rng(1984)
is_training = generator.random(len(table)) < 0.8
training, validation = table[is_training], table[~is_training]

# ---------------------------------------------------------------------------
# 2. Fit on the training pixels only, then look at what it learned
# ---------------------------------------------------------------------------
scaler = MinMaxScaler().fit(training[variables])
print(pd.DataFrame({"min": scaler.data_min_, "range": scaler.data_range_},
                   index=scaler.feature_names_in_))     # one row per variable

# ---------------------------------------------------------------------------
# 3. Apply to both sets: numpy arrays (pixels x variables), ready for the network
# ---------------------------------------------------------------------------
training_values = scaler.transform(training[variables])
validation_values = scaler.transform(validation[variables])
print(training_values.shape, training_values.dtype)     # (~450,000, 25) float32
print(training_values.min(), training_values.max())     # exactly 0 and 1

# Validation values can fall just outside 0-1, where they lie outside the
# training range. Count them per variable:
outside = ((validation_values < 0) | (validation_values > 1)).sum(axis=0)
print(pd.Series(outside, index=variables)[outside > 0])

# ---------------------------------------------------------------------------
# 4. A skewed variable: flow accumulation runs from ~1,000 to ~357 million,
#    so after scaling nearly every pixel sits at ~0 (see NOTES 2026-11-18)
# ---------------------------------------------------------------------------
flowacc = training_values[:, variables.index("flowacc_1km")]
print(np.quantile(flowacc, [0.5, 0.9, 0.99]))           # ~0.00001, 0.0005, 0.06

# ---------------------------------------------------------------------------
# 5. Save and load, as run.py and predict.py will
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as run_dir:          # a throwaway folder
    save_scaler(scaler, run_dir)
    loaded = load_scaler(run_dir)
    print(np.array_equal(loaded.transform(validation[variables]), validation_values))   # True

# Tidy up before exit, which avoids rioxarray's harmless shutdown message
del stack, template
import gc
gc.collect()
