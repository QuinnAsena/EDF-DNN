"""Tests for edf_dnn.data.grid, using tiny synthetic rasters (no real data needed).

Run with:  pixi run test
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from edf_dnn.data.grid import open_stack

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
