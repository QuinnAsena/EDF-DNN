# NOTES.md

Narrative log of decisions and findings. Stable reference is in
[CLAUDE.md](CLAUDE.md); open items are in [TODO.md](TODO.md).

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
