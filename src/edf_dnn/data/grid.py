"""Read the gridded driver rasters and put them on one common grid.

Contents
--------
- open_stack   : open a raster stack and keep the configured variables, by name.
- crop_to_grid : crop the historical stack to the future grid. Both are 1 km,
                 same CRS and already aligned, so this is a crop, not a resample.
- to_pixel_table : one row per pixel with cell_id and x/y, so every row can
                   always be traced back to a location.

Planned (phase 2)
-----------------
- Rebuild a float32 future stack from the future CSV. The current future
  raster is uint32 (values rounded), but the CSV's ID column is the 1-based,
  row-major cell number of the future grid, so each row can be placed exactly.

Temporary: the crop and the CSV rebuild are workarounds for the current
development data. On the new master grid both rasters should share one grid;
crop_to_grid then cuts nothing and only confirms that the grids match.
"""

import numpy as np
import pandas as pd
import rioxarray

def open_stack(raster_path, variables):
    """Open a multi-band GeoTIFF and keep only the named variables, in the given order.

    Parameters
    ----------
    raster_path : path to a GeoTIFF whose bands have names (band descriptions)
    variables   : list of band names to keep, e.g. ["burned01", "evergreen"]

    Returns
    -------
    An xarray DataArray with dimensions (band, y, x). The `band` coordinate
    holds the variable names, so one layer is `stack.sel(band="evergreen")`.
    Values are read lazily: nothing is loaded into memory until it is used.
    """
    # masked=True turns the file's nodata value into NaN, as terra does
    stack = rioxarray.open_rasterio(raster_path, masked=True)

    # rioxarray numbers the bands 1, 2, 3...; the names are stored separately
    # in the "long_name" attribute. A single-band file stores one string.
    band_names = stack.attrs.get("long_name")
    if band_names is None:
        raise ValueError(f"{raster_path} has no band names, so bands cannot be selected by name")
    if isinstance(band_names, str):
        band_names = (band_names,)
    band_names = list(band_names)

    # Fail loudly, listing every duplicated or missing name, rather than
    # continuing with the wrong layers. (A name listed twice in data.yaml
    # would otherwise give two copies of that band.)
    duplicated = sorted({name for name in variables if list(variables).count(name) > 1})
    if duplicated:
        raise ValueError(f"Variables listed more than once: {duplicated}")
    missing = [name for name in variables if name not in band_names]
    if missing:
        raise ValueError(f"Variables not found in {raster_path}: {missing}")

    # Label the bands with their names, then select by name
    stack = stack.assign_coords(band=band_names).sel(band=list(variables))

    # Some attributes can hold one value per ORIGINAL band: in our rasters
    # only the band names, in other files also statistics or scale/offset.
    # Cut those down to the selected bands so they stay correct.
    selected_positions = [band_names.index(name) for name in variables]
    stack.attrs = {
        key: (tuple(value[position] for position in selected_positions)
              if isinstance(value, tuple) and len(value) == len(band_names)
              else value)
        for key, value in stack.attrs.items()
    }
    return stack


def crop_to_grid(stack, template):
    """Cut a raster stack down to the area of a template raster on the same grid.

    The Python equivalent of terra::crop(stack, template), for grids whose
    cells already line up: values are only cut out, never resampled.

    Parameters
    ----------
    stack    : DataArray (band, y, x) to crop, e.g. the historical stack from open_stack
    template : DataArray on the target grid, e.g. rioxarray.open_rasterio(future_raster).
               Only its grid is used (CRS, cell size, position, size), never its values.

    Returns
    -------
    The stack cut to the template's rows and columns. It has the template's
    shape, transform and x/y coordinates, so a cell (row, column) is the same
    place on the ground in both. Still lazy: nothing is read from disk yet.
    """
    # Same projection? Like terra::same.crs(stack, template)
    if stack.rio.crs != template.rio.crs:
        raise ValueError(f"Different CRS: stack {stack.rio.crs}, template {template.rio.crs}")

    # Same cell size? Like comparing terra::res() of both
    if stack.rio.resolution() != template.rio.resolution():
        raise ValueError(
            f"Different cell sizes: stack {stack.rio.resolution()}, "
            f"template {template.rio.resolution()}"
        )

    # How many cells the template's top-left corner sits right of and below the
    # stack's top-left corner. A transform's .c and .f are the x and y of the
    # top-left corner (xmin and ymax of terra::ext()). Cell height is negative
    # because rows count downwards, which makes the row offset positive.
    cell_width, cell_height = stack.rio.resolution()
    column_offset = (template.rio.transform().c - stack.rio.transform().c) / cell_width
    row_offset = (template.rio.transform().f - stack.rio.transform().f) / cell_height

    # Aligned grids sit a whole number of cells apart. Anything else means the
    # cell edges don't line up, and the data would need terra::resample(), not
    # a crop. The tolerance (a millionth of a cell) allows for rounding error.
    off_grid = (abs(column_offset - round(column_offset)) > 1e-6
                or abs(row_offset - round(row_offset)) > 1e-6)
    if off_grid:
        raise ValueError(
            f"Grids don't line up: the template is offset by {column_offset} columns "
            f"and {row_offset} rows, not a whole number of cells"
        )
    column_offset, row_offset = round(column_offset), round(row_offset)

    # The template must lie entirely inside the stack
    if (column_offset < 0 or row_offset < 0
            or column_offset + template.rio.width > stack.rio.width
            or row_offset + template.rio.height > stack.rio.height):
        raise ValueError(
            f"The template runs past the edge of the stack: it starts at column "
            f"{column_offset}, row {row_offset} and is {template.rio.width} x "
            f"{template.rio.height} cells; the stack is {stack.rio.width} x {stack.rio.height}"
        )

    # Cut out the template's rows and columns, like R's stack[rows, cols].
    # isel selects by position, which is safe for rows and columns (their
    # order is fixed, unlike bands). Python counts from 0 and a slice's end
    # is excluded, so slice(259, 259 + 1081) is rows 259 to 1339.
    return stack.isel(
        y=slice(row_offset, row_offset + template.rio.height),
        x=slice(column_offset, column_offset + template.rio.width),
    )


def to_pixel_table(stack):
    """Turn a raster stack into a table with one row per pixel.

    The Python equivalent of
    terra::as.data.frame(stack, xy = TRUE, cells = TRUE, na.rm = TRUE).

    Parameters
    ----------
    stack : DataArray (band, y, x), e.g. the historical stack after crop_to_grid

    Returns
    -------
    A pandas DataFrame with columns cell_id, x, y, then one column per band,
    named after the band. cell_id is terra's cell number on the stack's grid:
    counted row by row from 1 at the top left. On the future grid it is the
    same number as the future CSV's ID. Pixels with a missing value in any
    band are dropped; the rest keep their cell_id, so rows are always matched
    by cell_id, never by row order.
    """
    # Read the values into memory: a numpy array (bands, rows, columns)
    values = stack.values
    band_count, row_count, column_count = values.shape

    # One row per pixel, one column per band. reshape walks the grid row by
    # row (terra's cell order); .T flips (bands x pixels) into (pixels x bands).
    # str() turns numpy's band-name type into plain Python text: sklearn only
    # records (and later checks) column names that are plain text.
    table = pd.DataFrame(
        values.reshape(band_count, row_count * column_count).T,
        columns=[str(name) for name in stack.band.values],
    )

    # Cell centre coordinates in the same row-by-row order. meshgrid repeats
    # the x values down every row and the y values across every column.
    x_grid, y_grid = np.meshgrid(stack.x.values, stack.y.values)
    table.insert(0, "cell_id", np.arange(1, row_count * column_count + 1))   # like terra::cells()
    table.insert(1, "x", x_grid.ravel())
    table.insert(2, "y", y_grid.ravel())

    # Drop pixels with a missing value in any band (na.rm = TRUE in terra),
    # then renumber the rows 0, 1, 2...: a pixel's identity is its cell_id,
    # not its row number
    return table.dropna().reset_index(drop=True)
