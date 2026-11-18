"""Tests for edf_dnn.data.grid, using tiny synthetic rasters (no real data needed).

Run with:  pixi run test
"""

import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401  (adds the .rio tools to xarray objects)
import xarray as xr
from rasterio.transform import from_origin

from edf_dnn.data.grid import crop_to_grid, open_stack, to_pixel_table

NODATA = -9999.0


def write_tiny_raster(path, band_names):
    """Write a 2 x 3 pixel GeoTIFF with named bands.

    Every value in band k equals k (band 1 is all 1s, band 2 all 2s...), so a
    test can tell which original band ended up where. The top-left pixel of
    every band is set to nodata, to check it becomes NaN.
    """
    height, width = 2, 3
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=len(band_names), dtype="float32",
        crs="EPSG:3857", transform=from_origin(0, 2000, 1000, 1000), nodata=NODATA,
    ) as destination:
        for band_number, name in enumerate(band_names, start=1):
            values = np.full((height, width), band_number, dtype="float32")
            values[0, 0] = NODATA
            destination.write(values, band_number)
            destination.set_band_description(band_number, name)


def test_selects_bands_by_name_in_requested_order(tmp_path):
    path = tmp_path / "tiny.tif"
    write_tiny_raster(path, ["burned01", "evergreen", "slope_1km"])

    stack = open_stack(path, ["slope_1km", "burned01"])

    assert list(stack.band.values) == ["slope_1km", "burned01"]
    # slope_1km was band 3 and burned01 band 1: the values prove the right
    # data came along with each name (mean skips the NaN pixel)
    assert float(stack.sel(band="slope_1km").mean()) == 3
    assert float(stack.sel(band="burned01").mean()) == 1
    # The stored band names now describe the selected bands only
    assert stack.attrs["long_name"] == ("slope_1km", "burned01")


def test_nodata_becomes_nan(tmp_path):
    path = tmp_path / "tiny.tif"
    write_tiny_raster(path, ["burned01", "evergreen"])

    stack = open_stack(path, ["evergreen"])

    assert np.isnan(stack.values[0, 0, 0])
    assert not np.isnan(stack.values[0, 1, 1])


def test_missing_variable_raises_and_names_it(tmp_path):
    path = tmp_path / "tiny.tif"
    write_tiny_raster(path, ["burned01", "evergreen"])

    with pytest.raises(ValueError, match="not_a_band"):
        open_stack(path, ["evergreen", "not_a_band"])


# ---------------------------------------------------------------------------
# crop_to_grid
# ---------------------------------------------------------------------------

def make_grid(left, top, height, width):
    """An in-memory one-band raster with 1 km cells and its top-left corner at (left, top).

    Each value is the cell's number, counted row by row from 0 at the top left
    (terra's cell numbers minus 1), so a test can see exactly which cells a
    crop kept.
    """
    cell_size = 1000
    values = np.arange(height * width, dtype="float32").reshape(1, height, width)
    # Coordinates are cell centres: half a cell in from the corner
    x = left + cell_size * (np.arange(width) + 0.5)
    y = top - cell_size * (np.arange(height) + 0.5)   # y decreases going down
    raster = xr.DataArray(values, dims=("band", "y", "x"), coords={"band": [1], "y": y, "x": x})
    return raster.rio.write_crs("EPSG:3857")


def test_crop_keeps_the_template_cells():
    stack = make_grid(left=0, top=5000, height=5, width=6)
    # The template starts 2 columns right of and 1 row below the stack's corner
    template = make_grid(left=2000, top=4000, height=2, width=3)

    cropped = crop_to_grid(stack, template)

    # Same grid as the template
    assert cropped.shape == template.shape
    assert cropped.rio.transform() == template.rio.transform()
    # The right cells: rows 1-2 and columns 2-4 of the stack. The top-left
    # cell of the crop is stack cell 1 * 6 + 2 = 8.
    assert cropped.values[0, 0, 0] == 8
    assert np.array_equal(cropped.values, stack.values[:, 1:3, 2:5])


def test_half_cell_shift_raises():
    stack = make_grid(left=0, top=5000, height=5, width=6)
    shifted = make_grid(left=2500, top=4000, height=2, width=3)   # 2.5 cells right

    with pytest.raises(ValueError, match="line up"):
        crop_to_grid(stack, shifted)


def test_template_past_the_edge_raises():
    stack = make_grid(left=0, top=5000, height=5, width=6)
    # Would need columns 4 to 6, but the stack only has columns 0 to 5
    too_far_right = make_grid(left=4000, top=4000, height=2, width=3)

    with pytest.raises(ValueError, match="edge"):
        crop_to_grid(stack, too_far_right)


# ---------------------------------------------------------------------------
# to_pixel_table
# ---------------------------------------------------------------------------

def test_pixel_table_rows_match_their_cells():
    # 2 rows x 3 columns; each value is the cell's number counted from 0
    stack = make_grid(left=0, top=2000, height=2, width=3).assign_coords(band=["cell_number"])

    table = to_pixel_table(stack)

    assert list(table.columns) == ["cell_id", "x", "y", "cell_number"]
    # cell_id counts from 1 like terra, so every value must be its cell_id - 1
    assert (table["cell_number"] == table["cell_id"] - 1).all()
    # Cell 5 is row 2, column 2 (counting from 1): its centre is 1.5 km
    # right of and 1.5 km below the top-left corner (0, 2000)
    cell_5 = table[table["cell_id"] == 5].iloc[0]
    assert (cell_5["x"], cell_5["y"]) == (1500, 500)


def test_pixel_table_drops_missing_and_keeps_cell_ids():
    one_band = make_grid(left=0, top=2000, height=2, width=3)
    stack = xr.concat([one_band, one_band], dim="band").assign_coords(band=["first", "second"])
    stack.values[0, 0, 1] = np.nan   # cell 2 missing in the first band only
    stack.values[1, 1, 0] = np.nan   # cell 4 missing in the second band only

    table = to_pixel_table(stack)

    # A pixel missing in any band is dropped; the others keep their cell_id
    assert list(table["cell_id"]) == [1, 3, 5, 6]
