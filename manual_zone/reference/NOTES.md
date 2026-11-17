# NOTES.md

Narrative log of what's been tried, what worked, and why. For stable
reference info (env, conventions, repo map) see [CLAUDE.md](CLAUDE.md); for
open action items see [TODO.md](TODO.md).

## Codebase restructure — 2026-06-26

The three complete MLP variants each had their own copy of model class, data
loading, and training loop (~200 lines per directory). These were extracted
into a shared `src/` module so that each variant is now a ~30-line `run.py`
config:

- `src/models.py` — all three architectures (`FeedforwardSingle`,
  `FeedforwardTwo`, `FeedforwardTapered`) with a shared constructor signature
  `(input_size, hidden_size, dropout)`
- `src/data.py` — `PREDICTORS` list (single source of truth), data loading,
  train/val/holdout splitting, downsampling, pos_weight, scaler fitting,
  tensor conversion
- `src/train.py` — `run_grid_search` (single-split, kept for reference) and
  `run_grid_search_cv` (k-fold, the active training entry point)

Per-variant `.qmd` files were updated to remove dead `eval: false` runpy
cells, fix imports (`sys.path.insert(0, '..')` + `from src.models import ...`),
and show the correct best-config plots.

## Stratified k-fold cross-validation — 2026-06-26

Replaced the single 80/20 split with proper k-fold CV (`k=5` default,
`StratifiedKFold` to preserve class ratio per fold). The key correctness
details:

- **Scaler fitted inside each fold**: prevents information about the validation
  fold's feature ranges from leaking into the training scaler.
- **Imbalance strategy applied inside each fold**: for downsampling this means
  each fold's training portion is balanced independently; for pos_weight the
  ratio is recomputed from each fold's training labels.
- **Held-out test set** (10%, split off before CV via `split_holdout`): the
  only data never seen during hyperparameter selection. Provides an unbiased
  final performance estimate.
- **Final model trained for `mean_stop_epoch` epochs** on the full training
  pool: the average number of epochs early stopping fired across folds, so no
  second validation split is needed for the final model.

Outputs: `gridsearch_results_cv.csv` (mean ± std per config),
`history_cv_h*_lr*_do*.pkl` (per-fold loss curves), `final_test_results.pkl`
(test-set metrics + checkpoint filename).

The old `run_grid_search` (single split) is still present in `src/train.py`
for reference; all `run.py` files now call `run_grid_search_cv`.

## MLP experiments

The target `burned01` is severely class-imbalanced (far more unburned than
burned pixels), so both complete approaches address that imbalance, by
different means:

- **Downsampling** (`mlp_single_lyr_downsampled/`, `mlp_two_lyr_downsampled/`): randomly downsample the majority (unburned) class in the *training* split only; validation stays at the real-world ratio.
- **Class weighting** (`mlp_single_lyr_weighted/`): keep all training rows, scale the loss via `pos_weight = num_neg / num_pos` in `BCEWithLogitsLoss`.

Best validation AUC per variant (from each `model_out/gridsearch_results.csv`):

| Variant | Best config | Val AUC | Val Acc |
|---|---|---|---|
| Two-layer, downsampled | h256 / lr1e-3 / do0.2 | **0.861** | 0.747 |
| Single-layer, downsampled | h256 / lr1e-3 / do0.1 | 0.796 | 0.706 |
| Single-layer, weighted | h256 / lr1e-3 / do0.1 | 0.798 | 0.709 |

**Finding**: going from one to two hidden layers gave the biggest jump in AUC
(~0.80 → 0.86). Class weighting did *not* outperform simple downsampling at
the same (single-layer) depth — despite addressing the same imbalance more
"properly" in theory, it landed at essentially the same AUC as downsampling.
Worth keeping in mind before defaulting to weighted loss for the CNN.

## Tapered MLP (paused, not abandoned-by-bug)

`mpl_two_lry_tapered/mlp_two_lyr_tapered.py` defines a tapered two-layer
architecture (hidden → hidden/2 → 1) but was never wired up to a training
script — no grid search, no checkpoints, no results. This looks like a
deliberate pause to try a third architecture idea, not a broken experiment.

## CNN spatial-pipeline exploration

Before any CNN architecture work could start, the raster data needed
historical/future alignment (matching CRS, extent, and resolution so a patch
sampled from one raster lines up with the other). The exploration visible in
`cnn_2d/` and git history went through several approaches, roughly in this
order:

1. **`spatial_test_gdal.sh`** — GDAL CLI tools (`gdalinfo`, `gdalwarp`) orchestrated from bash, extent/resolution parsed via `jq`. Proof of concept that the alignment is even possible with GDAL.
2. **`spatial_test_gdal.py`** — same idea via GDAL's Python bindings (`osgeo.gdal`) instead of shelling out — `gdal.Warp()` for the crop/resample.
3. **`spatial_test.py`** — moved to `rasterio` (higher-level than raw GDAL bindings): `rasterio.windows.from_bounds()` for cropping, `rasterio.warp.reproject()` for resampling (nearest for the `burned01` band, bilinear for continuous covariates), then converts to PyTorch tensors and saves via `joblib`. This is the most feature-complete pipeline so far.
4. **`spatial_crop.py`** (now deleted from the working tree, still in git history) — an earlier GDAL-subprocess-based cropping script, superseded by the rasterio/rioxarray work.
5. **`spatial_crop_rioxarray.py`** (new, untracked) — attempt to redo the same alignment using `rioxarray` (xarray-native, arguably more readable than raw rasterio windows/transforms — mirrors the `terra`-style workflow in `spatial_test.R`). **Currently non-functional** — see blocker below.

`spatial_test.R` exists alongside these as a `terra`-based reference
implementation of the same crop/resample logic, presumably to sanity-check
that the Python pipeline's output matches a trusted R workflow.

### Blocker — resolved 2026-06-24

`spatial_crop_rioxarray.py` carried the user's own comment at the top:

> "This script does not work due to package version issues between
> rioxarray, python 3.9 and numpy > 2."

Confirmed root cause by checking conda-forge's actual release history: with
`python=3.9.*` pinned, rioxarray's last compatible release is 0.15.0, which
needs `numpy<2`. numpy≥2 support only starts at rioxarray 0.20.0, which
requires `python>=3.12`. There is **no** rioxarray release compatible with
both `python=3.9` and `numpy>=2` simultaneously — `pixi.toml`'s old pin
(`rioxarray >=0.0.21,<0.0.22`, a 2020-era release with no numpy constraint
declared at all) was the solver's only way to satisfy `numpy>=2.0.2` +
`python=3.9.*` together, and it breaks at runtime because that code predates
numpy 2's API changes.

Cross-checked every other package in the stack (pytorch, rasterio, gdal,
scikit-learn, matplotlib, pandas, joblib, ipykernel, jupyter) against
conda-forge's python-3.13 builds — all have 3.13 builds, none require
`numpy<2`. So the fix applied: bumped `python` to `3.13.*` and `rioxarray` to
`>=0.22.0,<0.23` in `pixi.toml`, deleted and regenerated `pixi.lock`. Verified
end-to-end: `rioxarray.open_rasterio()` against the real
`drivers_historical_20092024.tif`/`drivers_NorESM2-MMssp370_20262040.tif`
rasters works, CRS matches, shapes load correctly.

**Side effect to know about**: because most other deps in `pixi.toml` are
pinned as open-ended ranges (`>=X,<Y`), the fresh solve under python 3.13
also picked up newer minor releases of everything else that had been capped
by python 3.9 compatibility: pytorch 2.7.1→2.12.1, scikit-learn 1.6.1→1.9.0,
numpy 2.0.2→2.5.0, pandas 2.3.1→2.3.3, rasterio 1.4.3→1.5.0, gdal
3.10.3→3.12.3, matplotlib 3.9.4→3.11.0. Verified an existing `.pt` checkpoint
(`mlp_single_lyr_downsampled/checkpoints/model_h256_lr0.001_do0.1.pt`) still
loads and runs a forward pass correctly under pytorch 2.12.1. Verified
`scaler.pkl` (fit under sklearn 1.6.1) still loads and transforms correctly
under sklearn 1.9.0, but sklearn raises `InconsistentVersionWarning` on
load — works today, flagged as a residual risk in [TODO.md](TODO.md) rather
than something to silently rely on.

Per the user: `rioxarray` was optional anyway since `spatial_test.R`
(terra) already produces the same aligned-raster output — this fix removes
that fallback dependency, it wasn't strictly necessary to unblock the CNN
work.

## Future: spatial block cross-validation

The k-fold CV currently implemented uses `StratifiedKFold` (random stratified
folds). This assumes observations are approximately independent, which
geographic pixels are not: nearby pixels share terrain, vegetation, fire
history, and climate. A burned pixel in fold 1 is spatially correlated with
its neighbours in fold 5, so the model can "cheat" by memorising local spatial
patterns across the fold boundary. The result is an optimistic CV AUC that
doesn't reflect how the model will perform when generalising to a genuinely
unseen geographic region.

**Spatial block CV** solves this by making folds contiguous geographic blocks
(e.g. a regular grid of rectangular tiles, or ecologically meaningful regions)
so train and val folds are spatially separated. The implementation would need
pixel coordinates (x/y or lat/lon) to assign each row to a block, then pass
those block IDs as the fold grouping to `sklearn.model_selection.GroupKFold`
instead of `StratifiedKFold`. Block size is a tuning choice: blocks need to
be large enough that autocorrelation doesn't bleed across the boundary, but
small enough that k≥5 distinct blocks exist with enough positive-class pixels
in each.

Worth implementing before citing final performance numbers, especially since
the downstream goal is spatial prediction over a future climate scenario — the
spatial generalisation error is the ecologically meaningful one.

Key reference: Roberts et al. (2017) "Cross-validation strategies for data
with temporal, spatial, hierarchical, or phylogenetic structure",
*Ecography* 40(8):913–929.
