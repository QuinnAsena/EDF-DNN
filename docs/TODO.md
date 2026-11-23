# TODO.md

Open items by phase. The full reasoning is in [plan.md](plan.md).
Each step: Claude explains and drafts, the user writes or approves, then
review. Every new function comes with its tests. See [NOTES.md](NOTES.md) for context.

## Phase 0-1: repo and environment

- [x] Folder skeleton, configs drafts, development data
- [x] pixi environment (CPU + declared cuda), editable package install, smoke test
- [x] Set git name/email in WSL, review the skeleton, first commit
- [x] Create the EDF-DNN GitHub repo and push
- [x] Delete geocomp-2025's local pixi env (3.1 GB) and its old Jupyter kernel
- [x] Copy development data into `data/` (replacing links to geocomp-2025)
- [x] Docs moved to `docs/` (+ repo description, glossary); `jobs/` renamed `cluster_jobs/`
- [ ] Archive geocomp-2025 on GitHub and delete its local folder (nothing in EDF-DNN depends on it)

## Phases 2-3, part 1: MLPs running again

Rebuild the old geocomp-2025 pipeline in the new structure, with random
stratified folds as before, so the old AUCs are a fair check (see NOTES.md,
2026-11-17).

- [x] `data/grid.py`: `open_stack` opens a raster and selects bands by name
- [x] `data/grid.py`: `crop_to_grid`, crop historical to the future grid
- [x] `data/grid.py`: `to_pixel_table`, stack -> pixel table with `cell_id` and x/y (replaces the old `load_historical`)
- [x] `train/metrics.py`: `compute_metrics` (average precision instead of trapezoid PR-AUC)
- [x] `data/normalise.py`: sklearn `MinMaxScaler`, with `save_scaler`/`load_scaler` (pickle in the run folder)
- [x] `data/batching.py`: `pixel_dataset` (a `TensorDataset`) + `batch_loader`, instead of a `PixelDataset` class
- [x] `train/loop.py`: `choose_device`, `train_one_epoch`, `predict_probabilities`, `fit` (early stopping on validation ROC AUC, patience 30, max 1000 epochs)
- [x] `data/splits.py`: `hold_out_test`, `assign_folds` (stratified), `downsample`
- [x] `train/run.py` (a): configs, run folder (`overwrite`, `config.yaml`, `git_commit.txt`), pixel table, test set and folds (`splits.csv`)
- [x] Rename the loaded historical pixel table `table` -> `historical_data` (run.py `main`, test_run, scratch scripts; verify_normalise.py by hand). Function arguments named `table` stay: they take any pixel table
- [x] `train/run.py` (b): one fold (scaler on the full training fold, then downsample), then the grid search; save per-pixel predictions (`cell_id`, label, probability) so curves and new metrics need no retraining
- [ ] `train/run.py` (c): final retrain on the whole pool for the median best epoch (log the mean too, to compare), scored once on the test set
- [x] MLP variant configs: `mlp_single`, `mlp_two_layer`, `mlp_tapered` (full grids), plus `dev_quick` for building code only
- [ ] Run the three MLP grids (a weekend on the laptop); compare single and two-layer with the old AUCs

## After the MLPs run: imbalance as an ensemble

- [ ] Repeated downsampling ("EasyEnsemble"): at least ~13 runs, each with all burned + a fresh random set of unburned pixels, predictions averaged (advice from Saverio, see NOTES 2026-11-19)
- [ ] Compare: one model with fresh unburned pixels every epoch; many small runs (e.g. 100 x 1,000 + 1,000)
- [ ] Correct averaged probabilities for the sampling ratio before mapping absolute burn probability

## Phases 2-3, part 2: spatial folds and prediction

- [ ] `data/blocks.py`: spatial block + fold map, saved once; replaces the random folds
- [ ] Rebuild a float32 future stack from the future CSV via `ID`
- [ ] `predict.py`: trained run -> future burn-probability GeoTIFF
- [ ] Future pixels have no labels: `pixel_dataset` and the prediction function need a no-label path

## Phase 4: patch CNN, then U-Net

- [ ] `PatchDataset` in `data/batching.py`; then replace `reprex_batches.py` with a real-data `verify_batching.py`

## Phase 5: cluster (PBS scripts in `cluster_jobs/`, Casper GPU test, grid search as job array)

## Phase 6: Quarto reports from `runs/`

- [ ] If a burned/unburned map is ever needed: TSS-vs-threshold curve from saved validation predictions (see NOTES 2026-11-19)

## Waiting on colleagues (new master-grid data)

- [ ] One `cell_id` + x/y carried in every product
- [ ] float32 rasters with band names and defined nodata
- [ ] Decadal summaries: which periods, and a burn label per period?
- [ ] Meaning of ysb = 100, and the reference date for years-since-burn
- [ ] Final variable set common to historical and future (and which skewed ones, like `flowacc_1km`, need a log before scaling: see NOTES 2026-11-18)

## Benched

- GRU / CNN-GRU (no annual time series)
