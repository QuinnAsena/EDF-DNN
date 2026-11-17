"""Read the gridded driver rasters and put them on one common grid.

Why this module exists
----------------------
Every model reads its inputs from here, so variable selection, cropping and
pixel coordinates are handled once. Bands are always selected by NAME, never
by position: band order differs between the historical and future rasters.

Planned contents (designed together in phase 2)
-----------------------------------------------
- Open a raster stack with rioxarray and select the configured variables by name.
- Crop the historical stack to the future grid. Both are 1 km, same CRS and
  already aligned, so this is a crop, not a resample.
- Expose x/y cell centres and a cell_id for every pixel, so tabular rows can
  always be traced back to a location.
- Rebuild a float32 future stack from the future CSV. The current future
  raster is uint32 (values rounded), but the CSV's ID column is the 1-based,
  row-major cell number of the future grid, so each row can be placed exactly.

Temporary: the last two points are workarounds for the current development
data and should become unnecessary once the new master-grid data arrive.
"""

import rioxarray


def open_stack(raster_path, variables):
    """Open a multi-band GeoTIFF and keep only the named variables, in the given order.

    The Python equivalent of terra::rast(raster_path)[[variables]].

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

    # Fail loudly, listing every missing name, rather than continuing
    # with the wrong layers
    missing = [name for name in variables if name not in band_names]
    if missing:
        raise ValueError(f"Variables not found in {raster_path}: {missing}")

    # Label the bands with their names, then select by name
    stack = stack.assign_coords(band=band_names).sel(band=list(variables))

    # Some attributes hold one value per ORIGINAL band (names, statistics,
    # scale/offset). Cut them down to the selected bands so they stay correct
    # if the stack is written back to a GeoTIFF.
    selected_positions = [band_names.index(name) for name in variables]
    stack.attrs = {
        key: (tuple(value[position] for position in selected_positions)
              if isinstance(value, tuple) and len(value) == len(band_names)
              else value)
        for key, value in stack.attrs.items()
    }
    return stack
