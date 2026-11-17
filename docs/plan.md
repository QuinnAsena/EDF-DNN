> Snapshot of the plan agreed on 2026-10-27, copied from Claude's plan file.
> Live progress is tracked in [TODO.md](TODO.md); this file is not updated.
> Later changes (e.g. `jobs/` renamed `cluster_jobs/`, docs moved to `docs/`)
> are logged in [NOTES.md](NOTES.md).

# Plan: a fresh repo for the EDF burn-risk deep learning project

## Context

The geocomp-2025 repo grew out of a course project and is now an EDF project. The
user wants a clean new repo built for the real goals: CNN first, then U-Net, running
locally at small scale now but ready for NCAR Casper/Derecho later. Time-series models
(GRU, CNN-GRU) are benched because no annual data exist. The scope is 2D spatial data:
one stack of variables per grid, possibly with a few decadal summaries.
All existing model outputs, checkpoints and derived rasters are treated as stale and
are not carried over. Input data is being rebuilt by colleagues on a common master
grid, and variable selection is still to be decided.

**Decisions (2026-10-27):** the new repo is called **EDF-DNN**. The new master-grid data
are still being processed, so the code base is developed on the existing historical and
future rasters, with minor adjustments expected when the new data land.

Working mode: Claude explains, proposes skeletons and reviews. The user writes or
approves each piece. Nothing below is built in one go.

### Answer 1: the pixel ID issue (verified 2026-10-27)

The historical *raster* is fine. Its CRS and geotransform define x/y for every cell, which
is why it plots correctly. Those coordinates are projected metres in Lambert Conformal
Conic, not lat/long, but they convert to lat/long exactly. The problem is only in the
*CSV tables* exported from the rasters. There are two separate problems.

1. **The historical CSV has no key at all.** When it was exported from the raster, the
   x/y and cell number were not written out as columns. Its rows can only be matched to
   pixels by row order. In terra, `as.data.frame(r, xy = TRUE, cells = TRUE)` keeps them.
2. **The one ID that exists is grid-specific.** The future table's `ID` is exactly
   terra's cell number in the future raster (row-major, starting at 1). The historical
   raster covers a larger extent, so the same pixel has a different cell number there.
   Example: future cell 436 is historical cell 500594. Adding a cell-number ID to the
   historical raster would therefore *not* cross-reference.

Linking by row order is unsafe because any step that drops NA rows, filters, sorts or
shuffles silently misaligns the rows, with no error raised. The historical CSV already
has fewer rows than the historical raster has valid burned01 cells, so some rows were
dropped somewhere.

**Fix:** the master grid solves the cross-referencing, provided one ID is defined once
on that grid and carried into every product. Ask for `cell_id` plus `x` and `y` cell
centres in the grid's CRS in every table. Coordinates are the universal fallback key
and are also what spatial block cross-validation needs.

### Answer 2: assessment of the current structure

What is good and worth keeping:
- Shared `src/` code with thin entry scripts.
- Scaler and imbalance handling done inside each fold, so no leakage.
- Fixed seeds, outputs separate from code, CLAUDE/NOTES/TODO documentation.

What will not scale to CNN and U-Net work:
- One directory per variant. Model × imbalance × variables × CV scheme multiplies folders.
- Hyperparameters live as constants inside `train.py`, so a result does not record what produced it.
- Every run overwrites `model_out/`.
- Full-batch training. CNNs need mini-batches from a DataLoader to fit in memory.
- The data path is tabular only. CNN patches need to be cut from the gridded rasters.
- `sys.path.insert` import hacks and hard-coded relative paths that break on a cluster.

The control the user values comes from explicit parameters, not from separate folders.
A config file per experiment gives *more* control: every setting is visible in one
place and is saved next to the results it produced.

## Recommended structure for the new repo

```
<new-repo-name>/
  pixi.toml              # default env (CPU) + cuda env for Casper/Derecho
  pyproject.toml         # makes src/<pkg> importable; no sys.path hacks
  configs/
    data.yaml            # data paths, variable list, block size, years
    experiments/
      mlp_baseline.yaml
      cnn_patch.yaml
      unet.yaml
  src/<pkg>/
    data/
      grid.py            # open master-grid raster stack by band NAME, expose x/y and cell_id
      blocks.py          # spatial block + fold assignment, saved once, shared by all models
      datasets.py        # PyTorch Datasets: PixelDataset (MLP), PatchDataset (CNN, U-Net)
      normalise.py       # stats from training blocks only, saved with each run
    models/              # mlp.py, cnn.py, unet.py
    train/
      loop.py            # train_one_epoch / evaluate, mini-batch, device-agnostic
      metrics.py         # one metrics function, used everywhere (AUC, avg precision, Brier...)
      run.py             # entry point: python -m <pkg>.train.run configs/experiments/x.yaml
    predict.py           # trained model -> future raster on the master grid
  jobs/                  # PBS scripts for Casper/Derecho, job arrays for grid search
  runs/                  # gitignored: runs/<experiment>/<timestamp>/{config, metrics, checkpoint}
  reports/               # Quarto book, reads only from runs/
  tests/                 # pytest on tiny synthetic grids
```

Key choices and why:

1. **One raster stack, two views.** All models read the same master-grid stack, with
   dimensions variable × y × x. The MLP gets single pixels as table rows and the CNN and
   U-Net get patches. Because the tabular view is cut from the raster in code, x/y and
   cell_id come with every row and the row-order problem cannot happen. Start by reading
   the colleagues' GeoTIFFs directly with rioxarray. Converting to chunked Zarr is an
   optional later step if reads are slow on the cluster's /glade filesystem.
   If decadal summaries arrive, each period's stack can be added as extra training
   samples rather than as a sequence, so no time-series model is needed.
2. **Spatial blocks defined once.** A saved block and fold map on the master grid is
   shared by every model. Patches must not straddle train and validation blocks, which
   is why this belongs in the data layer and not in each model.
3. **YAML configs read into Python dataclasses.** Simple, explicit, and typos fail
   loudly. No Hydra-style framework magic.
4. **Plain PyTorch with your own short training loop.** You can read and understand
   every line. Mini-batches and `device` handling from day one make the move to a GPU
   a config change. PyTorch Lightning is an option later if multi-GPU training is needed.
5. **Run folders that are never overwritten.** Each run saves its resolved config, git
   commit, normalisation stats, metrics and checkpoint together.
6. **Cluster-ready from the start.** Paths come from the config or environment
   variables, a pixi `cuda` environment exists alongside the CPU one, and PBS job
   scripts live in `jobs/`.

## Development data until the new data arrive

- **Historical:** read `drivers_historical_20092024.tif` directly, select bands by
  name, and crop it to the future grid, as `spatial_test.R` does. Both grids are 1 km,
  same CRS, and already aligned, so this is a crop, not a resample. x/y and cell_id
  come from the raster, so the CSV's missing key no longer matters.
- **Future:** the future raster is uint32 and rounded, so don't use it for values.
  Rebuild a float32 future stack from the future CSV instead: its `ID` is exactly the
  1-based cell number of the future grid, so every row can be placed back in its cell.
- The variable list stays the provisional 25 and lives in `configs/data.yaml`, so
  swapping it later is a one-file change.

## Questions for the colleague meeting (deferred until the new data are ready)

1. Will every product carry a single `cell_id` plus x/y on the master grid?
2. Will rasters be float32, with band names and a defined nodata value? The current
   future raster is uint32, which rounds fractions to 0 or 1 and FWI to integers.
3. **Decadal summaries.** If they exist, which periods, and does each period have its
   own burn label? Time-series models stay benched either way.
4. What does ysb = 100 mean, and from what date is years-since-burn measured?
5. Which variables will exist in both historical and future products?

## Phased steps

Each phase is a separate session: explain, draft together, user edits, review.

**This session covers phases 0 and 1 only, then stops for the user to review.**

0. **New repo, `~/projects/EDF-DNN`** inside WSL, not `/mnt/c`.
   - `git init`, the folder skeleton above with empty modules and docstrings, a
     `.gitignore` that excludes `data/`, `runs/`, `.pixi/` and rendered Quarto output.
   - A `data/` folder holding **symlinks** to the four existing files in the old repo,
     so 2.5 GB isn't copied. They become real files when the new data arrive.
   - Carry over `LICENSE`, and a new `CLAUDE.md` with the stable conventions: readable
     variable names, comments explaining the why, plan-first, Quarto not Jupyter.
     Fresh `NOTES.md` and `TODO.md` seeded from this plan.
   - Copy Claude's memory notes to the new project's memory folder.
   - The GitHub repo is created by the user, or by Claude with `gh` only after asking.
     Archiving geocomp-2025 on GitHub is left to the user. The old local copy, including
     its 3.1 GB `.pixi` env and stale outputs, is only deleted later and after asking.
1. **Environment.**
   - Update pixi itself to the latest release.
   - `pixi.toml` with the same core packages, pandas allowed to reach 3, pytest added,
     and the package installed in editable mode so `import` works from anywhere.
   - A separate `cuda` environment declared for later cluster use. It is not solved
     or installed locally since there is no GPU here.
   - Register a new Jupyter kernel for Quarto, named "EDF-DNN".
   - One trivial passing test so `pixi run pytest` is wired up.
   - First commit, after the user has looked at the skeleton.
2. **Data layer**, once the new data arrive. Name-based raster reader, variable list in
   config, block and fold map, normalisation. Tests on a tiny synthetic grid.
3. **Training core.** Datasets, loop, metrics, run folders. Port the MLP first as a
   pipeline check, since it is the simplest model and has old results to sanity-check.
4. **Patch CNN, then U-Net.** U-Net predicts every pixel of a tile at once, which suits
   mapping burn probability across the grid.
5. **Cluster.** PBS scripts, a first GPU test on Casper, grid search as a job array.
6. **Quarto reports** reading from `runs/`.

Benched: GRU and CNN-GRU. The structure leaves room for a `SequenceDataset` later if
annual data ever become available.

## Critical files

- Reference only, from the old repo: `src/models.py` (MLP classes), `src/train.py`
  (the per-fold scaler and imbalance logic in `run_grid_search_cv`), `cnn_2d/spatial_test.R`
  (correct name-based band selection).
- New: everything under the structure above, built phase by phase.

## Verification

- Phase 1: `pixi run python -c "import <pkg>"` works from any directory, and `pixi run pytest` passes.
- Phase 2: a test confirms `cell_id` and x/y round-trip between the pixel table and the
  source rasters, and that no pixel appears in both train and validation folds.
- Phase 3: the MLP on the new pipeline runs end to end on a small subset, and the run
  folder contains config, metrics and checkpoint.
- Phase 4 onward: the same small-subset run on CPU locally, then on a Casper GPU node
  with only a config change.
