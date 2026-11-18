# NOTES.md

Narrative log of decisions and findings. Stable reference is in
[CLAUDE.md](../CLAUDE.md); open items are in [TODO.md](TODO.md).

## Why a new repo (2026-10-27)

geocomp-2025 started as a course project and is now an EDF project. Its
structure (one folder per model variant, settings as constants in code,
full-batch training, tabular-only data) would not scale to CNN and U-Net work
or to cluster runs. All its outputs were stale. A fresh repo was simpler than
restructuring in place.

## Findings carried over from the geocomp-2025 review (2026-10-26)

- **Historical CSV has no pixel key.** The raster has coordinates, but x/y and
  cell number were not exported to the CSV, so its rows can only be matched to
  pixels by row order, which is unsafe.
- **Cell numbers are grid-specific.** The future CSV's `ID` is the 1-based,
  row-major cell number of the future raster (verified for all 563,783 cells).
  The historical raster has a larger extent, so the same pixel has a different
  cell number there (future 436 = historical 500594).
- **Grids are aligned.** Both are 1 km, same Lambert Conformal Conic CRS. The
  future grid sits at a whole-cell offset inside the historical grid
  (547 columns, 259 rows), so cropping aligns them exactly.
- **Future raster is uint32.** All 179 bands are whole numbers: fractions are
  0 or 1 and FWI is rounded. Use the future CSV for values.
- **Variable overlap.** Historical CSV has 149 columns, future 89, 58 shared.
  FFMC, DMC, DC, ISI, BUI and DSR exist only in historical.
- **No target leakage from years-since-burn.** Burned pixels are not
  disproportionately "recently burned". ysb is capped at 100, probably
  meaning "no recorded fire"; to confirm with the data producers.
- Old MLP results (single split, not CV): two-layer downsampled best,
  validation AUC about 0.86; single-layer about 0.80 with either imbalance method.

## Known harmless messages (2026-10-27)

- **"Error in sys.excepthook: Original exception was:"** with nothing after it,
  printed when a script that used `rioxarray.open_rasterio` exits. It comes
  from rioxarray/xarray's file cache being cleaned up during Python shutdown,
  and appears with plain rioxarray too. Data are unaffected. Deleting the
  stack and running `gc.collect()` before the script ends avoids it.
- **PendingDeprecationWarning "Use `@` matmul instead of `*`"** from inside
  rioxarray, triggered by a newer `affine` package. Filtered in pytest via
  `pyproject.toml`. Upstream will fix it; if rioxarray ever breaks on affine,
  this is the first suspect.

## Audit after a three-week break (2026-11-17)

Findings, all checked on the development data:

- **Cropping to the future grid loses no data.** All 77,005 burned cells and
  563,732 of the 563,735 valid `burned01` cells lie inside the future grid.
  The rest of the historical extent is empty (NaN). Burned share: 13.7%.
- **The cropped stack fits in memory.** 26 bands (target + 25 predictors) of
  1081 x 1329 cells is about 150 MB as float32.
- **All 25 configured variables exist** in the historical raster, the future
  raster and the future CSV.
- **The historical CSV is short by 705 rows** (563,030 rows against 563,735
  valid cells), another reason to read the raster instead.
- **Selecting bands by position would have gone wrong.** geocomp-2025's
  `cnn_2d/spatial_prep.py` took the first 26 historical and first 25 future
  bands. That picks e.g. `fwi_spring_*` and `ffmc_*` instead of `tcc`, the
  `ysb_*` and terrain variables: wrong layers for 13 of 25 historical and 8 of
  25 future predictors, with no error raised.

Decisions:

- **Back to running MLPs first.** Rebuild the old pipeline in the new structure
  with random stratified folds, as in geocomp-2025, before adding spatial
  folds and future prediction. Same splitting logic on nearly the same pixels
  means the old AUCs are a fair check of the port. Spatial block CV will
  likely score lower afterwards; that is the honest number, not a regression.
- **Nothing else needed from geocomp-2025.** `spatial_test.R` is replaced by
  `grid.py`, `verify_cv.py` is not needed, and the old
  `drivers_historical_aligned_to_future.tif` is rebuilt in memory by the crop.
- **Docs moved to `docs/`** (with a repo description and an ML glossary), and
  `jobs/` renamed `cluster_jobs/`. `manual_zone/` now holds only `reference/`
  and `scratch/`.

## Crop and pixel table on the development data (2026-11-17)

- **The pixel table cross-references the future data exactly.** On the cropped
  grid, `cell_id` is the future CSV's `ID`. Joined on it, the static terrain
  layers agree to about 1e-11 (elevation) and 1e-13 (slope) for all 563,162
  shared pixels; the same join shifted by one cell gives 561,187 mismatches.
- **563,475 complete pixels** (no missing value in any of the 26 bands), of
  563,732 with a valid `burned01`. The 257 dropped pixels (14 burned) each
  miss at least one predictor, mostly terrain or vegetation. 76,991 burned.
- **313 complete historical pixels have no row in the future CSV** (the two
  products have slightly different missing-value masks). No effect on
  training; matters only for which cells a future map can cover.
- **Why `crop_to_grid` checks alignment itself.** rioxarray's
  `rio.clip_box()` is a one-line `terra::crop()` look-alike and gives the same
  result on these aligned grids, but with a template shifted half a cell it
  silently returns 1,330 columns instead of 1,329, with no error.

## Metrics (2026-11-18)

- **One function, `compute_metrics`,** replaces the three copies in the old
  `train.py`. Same five numbers: ROC AUC, PR summary, Brier, recall and
  precision at 0.5.
- **Average precision replaces the trapezoid PR-AUC.** The PR curve is a
  staircase; straight lines between its corners misstate the area. On the
  four-pixel test example the two give 0.833 and 0.792. Old PR-AUC values are
  therefore not exactly comparable; ROC AUC is unchanged and is the number to
  compare with geocomp-2025.
- **Brier and recall/precision depend on how imbalance was handled.** A model
  trained on downsampled (50/50) data, or with `pos_weight`, predicts higher
  probabilities than the true 13.7% burn rate, which raises its Brier score
  and shifts recall/precision at 0.5. ROC AUC and average precision use only
  the ranking, so they compare fairly across imbalance methods. Brier
  baseline (always predicting the burned share): about 0.118.
