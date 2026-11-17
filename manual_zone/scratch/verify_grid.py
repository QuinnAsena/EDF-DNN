"""Interactive walkthrough of edf_dnn.data.grid.open_stack on the real data.

Run it line by line with Shift+Enter (console starts in the repo root), or all
at once with:  pixi run python manual_zone/scratch/verify_grid.py
"""

import numpy as np
import rasterio
import rioxarray
import yaml

from edf_dnn.data.grid import open_stack

# ---------------------------------------------------------------------------
# 1. The function's arguments, defined by hand
# ---------------------------------------------------------------------------
raster_path = "data/drivers_historical_20092024.tif"
variables = ["burned01", "tcc", "fwi_summer_mean"]

# ---------------------------------------------------------------------------
# 2. Step through the inside of open_stack, one line at a time
# ---------------------------------------------------------------------------
stack = rioxarray.open_rasterio(raster_path, masked=True)
print(stack)                           # dims, coordinates, attributes (like str() in R)
print(stack.shape)                     # (bands, rows, cols), like dim()
band_names = list(stack.attrs["long_name"])
print(band_names[:10])                 # band names are stored as an attribute
print(stack.band.values[:10])          # ...while the band coordinate is just 1, 2, 3...

missing = [name for name in variables if name not in band_names]
print("missing:", missing)             # should be an empty list

stack_named = stack.assign_coords(band=band_names)   # label bands with their names
stack_selected = stack_named.sel(band=variables)     # then select by name
print(stack_selected.band.values)

# ---------------------------------------------------------------------------
# 3. Now call the real function and compare
# ---------------------------------------------------------------------------
result = open_stack(raster_path, variables)
print(result)
print(result.attrs["long_name"])       # trimmed to the selected bands

# Independent check: read each band straight from the file with rasterio
with rasterio.open(raster_path) as source:
    for name in variables:
        direct = source.read(source.descriptions.index(name) + 1)
        same = np.array_equal(result.sel(band=name).values, direct, equal_nan=True)
        print(f"{name:16s} matches rasterio: {same}")

# ---------------------------------------------------------------------------
# 4. Same thing with the variable list from the config file
# ---------------------------------------------------------------------------
config = yaml.safe_load(open("configs/data.yaml"))
config_variables = [config["target"]] + config["variables"]
full_stack = open_stack(f"{config['data_dir']}/{config['historical_raster']}", config_variables)
print(full_stack.shape)                # (26, 1547, 1929)

# A quick map. In the interactive console this opens a plot window;
# when run as a whole script, the plot is skipped unless you add plt.show().
full_stack.sel(band="tcc").plot()

# Tidy up before exit, which avoids rioxarray's harmless shutdown message
del stack, stack_named, stack_selected, result, full_stack
import gc
gc.collect()   # make Python release the files now, not during shutdown
