"""Interactive walkthrough of edf_dnn.data.grid (open_stack, crop_to_grid, to_pixel_table) on the real data.

Run it line by line with Shift+Enter (console starts in the repo root), or all
at once with:  pixi run python manual_zone/scratch/verify_grid.py
"""

import numpy as np
import pandas as pd
import rasterio
import rioxarray
import yaml

from edf_dnn.data.grid import crop_to_grid, open_stack, to_pixel_table

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

# import matplotlib.pyplot as plt    # not "import matplotlib as plt"
# full_stack.sel(band="tcc").plot()
# plt.show()

# ---------------------------------------------------------------------------
# 5. crop_to_grid: cut the historical stack down to the future grid
# ---------------------------------------------------------------------------
# The function's arguments, defined by hand
stack = full_stack
template = rioxarray.open_rasterio(f"{config['data_dir']}/{config['future_raster']}")

# Step through the inside of crop_to_grid
print(stack.rio.crs == template.rio.crs)        # same projection: True
print(stack.rio.resolution(), template.rio.resolution())   # (1000.0, -1000.0) for both
print(stack.rio.transform())                    # c and f: x and y of the top-left corner
print(template.rio.transform())

cell_width, cell_height = stack.rio.resolution()
column_offset = (template.rio.transform().c - stack.rio.transform().c) / cell_width
row_offset = (template.rio.transform().f - stack.rio.transform().f) / cell_height
print(column_offset, row_offset)                # 547.0 259.0: whole cells, so the grids line up

# Now call the real function and compare
cropped = crop_to_grid(stack, template)
print(cropped.shape)                            # (26, 1081, 1329): the future grid's size
print(cropped.rio.transform() == template.rio.transform())   # True

# Independent check: every cell centre of the crop matches the future file's own coordinates
print("x match:", np.array_equal(cropped.x.values, template.x.values))
print("y match:", np.array_equal(cropped.y.values, template.y.values))

# The crop loses no data: same number of burned cells before and after (77,005)
print("burned before:", int((full_stack.sel(band="burned01") == 1).sum()))
print("burned after: ", int((cropped.sel(band="burned01") == 1).sum()))

# ---------------------------------------------------------------------------
# 6. to_pixel_table: one row per pixel, like terra::as.data.frame(xy = TRUE, cells = TRUE)
# ---------------------------------------------------------------------------
# Step through the inside of to_pixel_table
values = cropped.values                         # reads the 26 bands into memory (~150 MB)
print(values.shape)                             # (26, 1081, 1329): bands, rows, columns
band_count, row_count, column_count = values.shape
x_grid, y_grid = np.meshgrid(cropped.x.values, cropped.y.values)
print(x_grid[:2, :3])                           # x changes along a row...
print(y_grid[:2, :3])                           # ...y changes down a column

# Now the real function
historical_data = to_pixel_table(cropped)
print(historical_data.head())                   # cell_id, x, y, burned01, evergreen, ...
print(len(historical_data))                     # 563,475 complete pixels out of 1,436,649 cells
print(historical_data["burned01"].value_counts())  # like table() in R

# Cross-reference with the future CSV. Elevation and slope don't change over
# time, so joining on cell_id = ID must give identical values, if the IDs match.
future = pd.read_csv(f"{config['data_dir']}/{config['future_table']}",
                     usecols=["ID", "elevation_1km", "slope_1km"])
joined = historical_data.merge(future, left_on="cell_id", right_on="ID",  # like dplyr::inner_join()
                               suffixes=("_historical", "_future"))
print(len(joined))                              # pixels found in both tables
for name in ["elevation_1km", "slope_1km"]:
    difference = (joined[f"{name}_historical"] - joined[f"{name}_future"]).abs()
    print(f"{name}: largest difference {difference.max():.2g}")   # ~0: every row lines up

# Control: shift the IDs by one cell and the same check fails almost everywhere
shifted = historical_data.assign(cell_id=historical_data["cell_id"] + 1)
shifted = shifted.merge(future, left_on="cell_id", right_on="ID", suffixes=("_historical", "_future"))
mismatches = (shifted["elevation_1km_historical"] - shifted["elevation_1km_future"]).abs() > 0.001
print("off by one cell, elevations that differ:", int(mismatches.sum()))

# Tidy up before exit, which avoids rioxarray's harmless shutdown message
del stack, stack_named, stack_selected, result, full_stack, template, cropped
import gc
gc.collect()   # make Python release the files now, not during shutdown
