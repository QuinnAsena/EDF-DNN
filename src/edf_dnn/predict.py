"""Apply a trained run to the future grid and write a burn-probability raster.

Planned (phase 3/4): load a run folder's config, scaling statistics and
checkpoint, predict for every cell of the future stack, and write a float32
GeoTIFF on the same grid, so results open directly in R/terra or QGIS.
"""
