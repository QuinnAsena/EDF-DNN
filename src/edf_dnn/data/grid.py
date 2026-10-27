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
