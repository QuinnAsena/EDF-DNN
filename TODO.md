# TODO.md

Open items by phase. Each phase: Claude explains and drafts, the user writes
or approves, then review. See [NOTES.md](NOTES.md) for context.

## Phase 0-1: repo and environment

- [x] Folder skeleton, configs drafts, symlinked development data
- [x] pixi environment (CPU + declared cuda), editable package install, smoke test
- [ ] Set git name/email in WSL, review the skeleton, first commit
- [ ] Create the EDF-DNN GitHub repo and push
- [ ] Archive geocomp-2025 on GitHub; later delete its local copy (3.1 GB env + stale outputs)

## Phase 2: data layer (on development data)

- [ ] `data/grid.py`: open stacks, select bands by name, crop historical to future grid
- [ ] x/y and cell_id for every pixel; rebuild float32 future stack from the CSV via ID
- [ ] `data/blocks.py`: spatial block + fold map, saved once
- [ ] Tests on a tiny synthetic grid

## Phase 3: training core

- [ ] `train/metrics.py` (one function; average precision instead of trapezoid PR-AUC)
- [ ] `data/datasets.py` PixelDataset, `data/normalise.py`
- [ ] `train/loop.py` small functions; `train/run.py` with run folders and `overwrite`
- [ ] Port the MLP as the first end-to-end check; `predict.py` to a GeoTIFF

## Phase 4: patch CNN, then U-Net

## Phase 5: cluster (PBS scripts, Casper GPU test, grid search as job array)

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
