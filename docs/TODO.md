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
- [ ] `data/grid.py`: `crop_to_grid`, crop historical to the future grid
- [ ] `data/grid.py`: stack -> pixel table with `cell_id` and x/y (replaces the old `load_historical`)
- [ ] `train/metrics.py` (one function; average precision instead of trapezoid PR-AUC)
- [ ] `data/normalise.py` (scaler fitted per fold, on training cells only)
- [ ] `data/datasets.py`: `PixelDataset`
- [ ] `train/loop.py`: train one epoch, evaluate, early stopping
- [ ] `train/run.py`: held-out test set + stratified k-fold, run folders with `overwrite`
- [ ] Old MLP variants as experiment configs; compare with the old AUCs

## Phases 2-3, part 2: spatial folds and prediction

- [ ] `data/blocks.py`: spatial block + fold map, saved once; replaces the random folds
- [ ] Rebuild a float32 future stack from the future CSV via `ID`
- [ ] `predict.py`: trained run -> future burn-probability GeoTIFF

## Phase 4: patch CNN, then U-Net

## Phase 5: cluster (PBS scripts in `cluster_jobs/`, Casper GPU test, grid search as job array)

## Phase 6: Quarto reports from `runs/`

## Waiting on colleagues (new master-grid data)

- [ ] One `cell_id` + x/y carried in every product
- [ ] float32 rasters with band names and defined nodata
- [ ] Decadal summaries: which periods, and a burn label per period?
- [ ] Meaning of ysb = 100, and the reference date for years-since-burn
- [ ] Final variable set common to historical and future

## Benched

- GRU / CNN-GRU (no annual time series)
- Tapered MLP (trivial to add once the training core exists)
