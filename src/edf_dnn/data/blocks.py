"""Spatial blocks and cross-validation folds on the grid.

Why this module exists
----------------------
Neighbouring pixels share terrain, vegetation and fire history, so random
k-fold splits let a model "see" the validation area through its neighbours
and report optimistic scores. Spatial block CV assigns whole rectangular
blocks of the grid to folds instead (Roberts et al. 2017, Ecography).

The fold map is computed once and saved, and every model uses the same map,
so results are comparable across MLP, CNN and U-Net.

Planned contents (phase 2)
--------------------------
- Assign each grid cell to a block from its x/y and a block size in km.
- Assign blocks to k folds, checking each fold has enough burned pixels.
- Save/load the fold map as a raster aligned to the grid.
- For CNN patches: identify patches that straddle a fold boundary.
"""
