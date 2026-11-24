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
  `train.py`. The old five numbers (ROC AUC, PR summary, Brier, recall and
  precision at 0.5) plus burned share, Brier skill score, log loss,
  specificity and TSS. Accuracy, kappa, F1 and MCC left out: accuracy and
  kappa depend on the burned share (always predicting "unburned" is 86%
  accurate), and F1/MCC repeat what recall, precision and TSS already show.
- **It refuses one-class labels and out-of-range probabilities.** sklearn 1.9
  returns NaN ROC AUC with only a warning when a set has no burned pixels
  (possible with spatial folds); logits passed without the sigmoid would
  leave ROC AUC unchanged but make Brier and recall meaningless.
- **Any other metric can be added later without retraining**, as long as the
  runs save per-pixel predictions (`cell_id`, label, probability). Only the
  metrics used during training (early stopping, picking the best
  hyperparameters) must be fixed before the runs.
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

## Scaling (2026-11-18)

- **sklearn's `MinMaxScaler`, saved as a pickle, as in geocomp-2025.**
  `normalise.py` only holds `save_scaler`/`load_scaler`, so the file name
  (`scaler.pkl` in the run folder) is defined once. A home-made version
  (statistics as CSV, plus a "standard" option) was drafted and dropped as
  overkill: fitted on a DataFrame, `MinMaxScaler` already refuses a table
  whose columns differ or come in a different order; `pixi.lock` pins the
  sklearn version, so the pickle loads on laptop and cluster alike (another
  version gives an `InconsistentVersionWarning`, not silence); and
  `data_min_`/`data_range_` give the numbers for raster stacks later.
- **That name check only works with plain-text column names.** xarray hands
  band names over as numpy's `str_` type, and sklearn then records no names
  and checks nothing, without a warning: reordered columns went through.
  Found in the real-data walkthrough; `to_pixel_table` now converts the names
  with `str()`, and a test fits a scaler on its output.
- **`flowacc_1km` is useless as scaled.** It runs from about 1,000 to 357
  million; after min-max scaling half the pixels are below 0.00001, so the
  network sees almost a constant. Options: log-transform it first, or drop it
  (`twi_1km` is already built from the log of flow accumulation). Decide with
  the final variable set.
- **Order of scaling and downsampling, to settle in `run.py`.** The old code
  downsampled the training fold first and fitted the scaler on what was left.
  Fitting on the full training fold is steadier, and gives the same scaling
  whichever imbalance method is used.
- **pandas gotcha, for later code:** assigning several columns at once
  (`df[names] = other`) matches columns by position, not name.

## Datasets and mini-batches (2026-11-19)

- **No `PixelDataset` class.** PyTorch's built-in `TensorDataset` does the
  job: `pixel_dataset` scales the predictors and bundles features, labels and
  `cell_id` (so predictions are matched back by `cell_id`, not row order).
  The hand-written Dataset comes with the CNN's `PatchDataset`.
- **Fetch whole batches, not single pixels.** A plain `DataLoader` fetches
  pixel by pixel and regroups: 4.9 s per epoch just for fetching 450,000
  pixels, more than the training itself (7.2 s per epoch in total with a
  two-layer MLP on the laptop CPU). `batch_loader` hands the DataLoader whole
  batches of row numbers (`BatchSampler`, `batch_size=None`): 0.4 s. On the
  real training pixels: 111 batches of 4096 per epoch, fetched in 0.42 s.
- **Why shuffle.** The pixel table is in map order, so unshuffled batches
  would each come from one strip of the map. Shuffled batches are random
  mixes of regions and of burned/unburned pixels, redrawn every epoch.

## Imbalance as an ensemble of downsampled runs (recorded 2026-11-19, for after the MLPs run)

Advice from Saverio Mancino (geocomp-2025 course, 2025-11-29) on the old
set-up: random training/validation split, training pixels downsampled to
50/50, validation left at the real ratio.

- **The set-up is sound.** Downsample only the training pixels; keep
  validation (and test) at the real ratio, as the map will be.
- **Better: repeat it and average.** Train many times, each with a different
  random sample of unburned pixels, and average the models' predictions.
  Together the runs use all unburned pixels and capture their variability
  instead of discarding most of them; each run is balanced, so there is no
  built-in bias, which also helps when explaining the model.
- **How many runs: more than double the class ratio.** The old training set
  had 61,540 burned and 388,884 unburned pixels (about 6.3 to 1), so at
  least about 13 runs, each with all burned pixels plus as many random
  unburned ones.
- **Cheaper alternative:** many small balanced runs that subsample the burned
  pixels too (e.g. 100 runs of 1,000 + 1,000). Also captures the variability
  among burned pixels, at the cost of less data per run.
- `pos_weight` (weighting the loss instead) was tried in geocomp-2025 without
  a clear gain.

Notes for when we get there:

- This is "EasyEnsemble": Liu, Wu & Zhou (2009), IEEE Transactions on
  Systems, Man, and Cybernetics B 39: 539-550,
  https://doi.org/10.1109/TSMCB.2008.2007853
- Mini-batches make it affordable: one downsampled run takes minutes on the
  laptop CPU, so 13+ runs is a small job array on the cluster.
- A single-model variant worth comparing: draw a fresh random set of unburned
  pixels every epoch, so one model sees them all over its training.
- Models trained on 50/50 data predict probabilities above the true burn
  rate, and so does their average. Rankings (ROC AUC, average precision) are
  unaffected; maps of absolute probability (and the Brier score) need a
  correction for the sampling ratio.

## Training curve with mini-batches (2026-11-19)

One run on the development data: two-layer MLP (hidden 128, dropout 0.1,
Adam 1e-3), training pixels downsampled to 50/50, random 80/20 split,
batches of 4096, 600 epochs (about 10 minutes on the laptop CPU).

- **Mini-batches vs full-batch** (a separate run on all 450,756 training
  pixels, not downsampled). Same model, same number of weight updates (111),
  same validation AUC (0.72); 1 mini-batch epoch took 2.7 s against 447 s
  for 111 full-batch epochs.
- **No overfitting within 600 epochs on random folds.** Validation AUC rose
  steadily: 0.81 at epoch 50, 0.87 at 300, 0.89 at 600 (the old two-layer
  downsampled result was about 0.86). Random folds flatter this: validation
  pixels sit next to training pixels. Expect an earlier plateau with
  spatial folds.
- **Validation loss is a poor early-stopping signal with downsampling.** It
  rose from one epoch to the next in 50% of epochs. The model is trained on
  50/50 but validated at the real 13.7%, so the loss mostly tracks how
  inflated the probabilities are, which wobbles, while the ranking improves
  steadily. Replaying stopping rules on the logged curve:

  | Watching | Patience | Stops at | Keeps epoch | Val AUC kept |
  |---|---|---|---|---|
  | val loss | 10 | 45 | 35 | 0.793 |
  | val loss | 20 | 96 | 76 | 0.821 |
  | val loss | 50 | 374 | 324 | 0.871 |
  | val AUC | 10 | 408 | 398 | 0.880 |
  | val AUC | 20+ | did not stop | 599 | 0.892 |

  Validation AUC fell from one epoch to the next in 38% of epochs.
- **Decided: early stopping watches validation ROC AUC** (also the metric
  for picking hyperparameters), `patience: 30`, `max_epochs: 1000` as a
  time cap. Run time is not a constraint (overnight or weekend runs are
  fine). With all validation metrics logged every epoch, one epoch takes
  about 1.1 s on the laptop CPU, so about 19 minutes per fold at 1000.

## Threshold for recall, precision and TSS (2026-11-19)

- **Not a hyperparameter.** The threshold changes nothing in training, early
  stopping (ROC AUC), hyperparameter choice or the probability maps; only
  recall, precision, specificity and TSS. It can be chosen afterwards from
  the saved validation predictions, without retraining.
- **0.5 is the right default for a model trained on 50/50 data.** On a
  downsampled-trained two-layer MLP (100 epochs, validation ROC AUC 0.83),
  TSS peaked at exactly 0.50 (0.493; 0.492 at 0.55; 0.405 at 0.30, 0.415 at
  0.70). In theory, 0.5 on the balanced scale is the burned share (13.7%) on
  the real scale: p_real = b p / (b p - p + 1) with b = burned/unburned =
  0.158 gives 0.137 at p = 0.5. Liu et al. (2005), Ecography 28: 385-393,
  https://doi.org/10.1111/j.0906-7590.2005.03957.x, compare threshold rules
  for species distribution models.
- The same model's mean predicted probability was 0.40 against a burned share
  of 0.136: the inflation noted in the ensemble entry above.

## Review before run.py (2026-11-19)

Full read of docs, configs and memory, plus an independent code review. No
correctness bugs in the implemented code. Fixed: a duplicated glossary row;
two comments in `datasets.py` that misdescribed `batch_size=None` and the
future `PatchDataset`; `open_stack` now refuses a variable listed twice; a
test that would have missed scaler leakage in `pixel_dataset`; smaller
wording fixes. For the `run.py` design:

- **Seed per fold and per hyperparameter setting.** Model initialisation,
  dropout and shuffling all draw from torch's one global random-number
  stream, and even evaluation loaders consume from it, so otherwise a
  setting's result depends on what ran before it.
- **Final retrain has no validation set,** so `fit` (which needs one) does not
  fit; train a fixed number of epochs from the folds' best epochs instead.
- Models are looked up by class name from the config, but
  `edf_dnn.models` does not expose the classes yet.
- Only the final model's scaler needs saving (per-fold results are kept as
  per-pixel predictions); create the run folder before saving into it.
- Use positions (`.iloc`), not labels, with `StratifiedKFold` indices.
- Paths in `data.yaml` must resolve from the repo root, not the working
  directory (cluster jobs start elsewhere).
- A fold's best-epoch validation AUC is slightly optimistic (the best of many
  epochs on that fold); only the held-out test score is unbiased.

## Decisions after the review (2026-11-19)

- **`pos_weight` dropped.** In geocomp-2025 it gave the same validation AUC
  as downsampling (single-layer: 0.798 weighted, 0.796 downsampled), and
  downsampling trains about 3.7x faster per epoch and is the base of the
  planned ensemble. Every training fold is downsampled; the `imbalance:`
  config key is gone.
- **MLP variants:** single-layer and two-layer (to compare with the old AUCs)
  plus the tapered MLP, which existed in geocomp-2025 but was never trained.
  The MLPs are a stepping stone; the real gains are expected from the CNN
  and U-Net.
- **Renames:** `predict` -> `predict_probabilities` (it serves validation,
  test and future pixels alike, and `edf_dnn.predict` is the future-map
  module); MLP layers `fc1`/`fc2`/`fc3` -> `hidden_layer`,
  `first_hidden_layer`, `second_hidden_layer`, `output_layer`.
- **Threshold stays 0.5** (see "Threshold" above).

## run.py, step (a) (2026-11-19)

- **Decided:** the scaler is fitted on the full training fold, then the fold
  is downsampled (the old code downsampled first): min and max then come
  from about 405,000 pixels instead of 111,000. The final model trains for
  the median best epoch across folds (the old code used the mean stop epoch,
  which runs `patience` epochs past the best); `cv_results.csv` will log the
  mean too, to compare.
- **Splits are shared by all experiments** (`test_fraction`, `n_folds`,
  `random_seed` in `data.yaml`) and saved per run in `splits.csv`. On the
  development data: 56,348 test pixels, a pool of 507,127 in 5 folds of
  about 101,400; burned share 0.137 in every one.
- **`dev_quick.yaml` is for building code only:** the old best two-layer
  setting with 20 epochs, writing to its own `runs/dev_quick/`. The real
  configs hold the full geocomp-2025 grid, which contains the old best
  settings, so the full runs give the comparison with the old AUCs.
- `mlp_baseline.yaml` was renamed `mlp_two_layer.yaml`; a test checks that
  every config's `name` matches its file name, so two configs can never
  write to the same run folder.

## run.py, step (b) (2026-11-23)

- **What it does:** `grid_settings` expands the config lists with sklearn's
  `ParameterGrid` (12 settings for the MLP grids); `train_one_fold` is
  `walkthrough_mlp.py` sections 3-7 as a function; `cross_validate` runs
  every setting on every fold. The architecture is looked up by its config
  name in `edf_dnn.models` (`get_architecture`).
- **Seed reset before every model** (`torch.manual_seed` inside
  `train_one_fold`), so each setting and fold gives the same result whatever
  runs before it, and can be rerun alone (later: one cluster job each).
- **Files:** `cv_folds.csv` (one row per setting and fold, including
  `epochs_run`, to spot runs that hit `max_epochs`), `cv_results.csv` (one
  row per setting: mean and sd of every validation metric, median and mean
  best epoch, `folds_done`; best mean ROC AUC first), `cv_history.csv`
  (every epoch) and `cv_predictions.parquet` (every validation pixel, per
  setting: about 5 MB per setting).
- **Parquet, not CSV, for predictions** (507,127 rows per setting): smaller,
  faster, keeps types, and R reads it with `arrow::read_parquet()`.
  `pyarrow` added to `pixi.toml`.
- **Crash guard:** the cv_ files are rewritten after every fold, so a crash
  loses only the fold in progress. No resume yet (later, with job arrays).
- **Tests and mutants:** six realistic bugs (no reseed, folds swapped,
  scaler fitted on validation, scaler fitted after downsampling, saving
  only at the end, best setting sorted last) are each caught by a test.
- **dev_quick end to end on the real data:** 3.5 minutes; fold 1 ROC AUC
  0.781, identical to `walkthrough_mlp.py`, so run.py reproduces the hand
  version. About 1.8 s per epoch at hidden size 256 (each epoch also scores
  101,400 validation pixels). If early stopping never triggers, a fold takes
  up to ~30 min (hidden 256) or ~19 min (hidden 128), so the two-layer grid
  could take about a day; the three grids together may not fit one weekend.
- **Rename:** `data/datasets.py` -> `data/batching.py` (and its test file):
  "datasets" read like the input data, not PyTorch's Dataset objects.

## Training checks against known answers (2026-11-24)

`manual_zone/scratch/verify_training.py`, fold 1, two-layer MLP (hidden 256,
dropout 0.2, learning rate 1e-3), after Karpathy's "A Recipe for Training
Neural Networks" (2019, http://karpathy.github.io/2019/04/25/recipe/):

- **Loss before training:** 0.695 (expected -log(0.5) = 0.693); mean
  probability 0.48.
- **Memorising 64 pixels:** training loss 0.70 -> 0.02 (100 epochs) ->
  0.0001 (500); ROC AUC 1.0. Gradients and optimizer are connected.
- **Baseline:** logistic regression (unpenalised, like `glm`) on the same
  scaled, downsampled pixels: validation ROC AUC 0.712.
- **Early stopping (patience 2):** best epoch 84 = the history's highest
  AUC, stopped at 86 = best + patience, and the restored weights reproduce
  epoch 84's AUC exactly (0.8415; the last epoch had 0.8406).
- **Still improving at 200 epochs:** with patience 5 the same model never
  stopped; at epoch 200 its validation ROC AUC was 0.874, already above
  geocomp-2025's 0.861 (single split). So the real runs (patience 30) will
  probably run long, and some may reach the 1000-epoch cap: check
  `epochs_run` and `best_epoch` in `cv_folds.csv`. Worst case for the
  two-layer grid is about a day.
- **Run folder checks** (`manual_zone/scratch/verify_run.py`, on
  `runs/dev_quick/`): every prediction sits in its own fold, no test pixel
  was predicted, labels match `burned01` by `cell_id`, metrics recomputed
  from `cv_predictions.parquet` equal `cv_folds.csv`, `cv_results.csv`
  equals the fold means and sds, and `cv_history.csv` agrees on best
  epochs. On a corrupted copy (two labels swapped, one best epoch changed)
  the label, metric and history checks all fail, as they should. Rerun it on
  each real grid once it finishes.
- **Rename:** `pixel_dataset` -> `table_to_tensors` (in `data/batching.py`).
  The old name said what kind of object comes back (a PyTorch Dataset) but
  not what the function does: pixel table in, scaled tensors out. The earlier
  entries above keep the old name.
