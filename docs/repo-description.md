# Repo description

Where everything lives, and what each part is for. For progress see
[TODO.md](TODO.md); for setup and commands see [runbook.md](runbook.md).

## The map

```
EDF-DNN/
├── CLAUDE.md, README.md    project description; README is what GitHub shows
├── pixi.toml, pixi.lock    the environment: packages asked for / exact versions installed
├── pyproject.toml          makes src/edf_dnn importable as `edf_dnn`; pytest settings
├── configs/                settings, not code
│   ├── data.yaml           file names, the variables, fold settings (shared by all runs)
│   └── experiments/        one YAML per experiment: model, hyperparameters, imbalance method
├── src/edf_dnn/            all the real code
│   ├── data/               rasters -> pixels or patches -> tensors
│   ├── models/             network architectures
│   ├── train/              training loop, validation, metrics, run.py
│   └── predict.py          trained model -> future burn-probability GeoTIFF
├── tests/                  automated checks (`pixi run test`)
├── benchmarks/             training-speed benchmark on synthetic data, and its results
├── runs/        (gitignored)  outputs: one folder per training run
├── cluster_jobs/           job scripts for the NCAR cluster
├── reports/                Quarto .qmd reports
├── data/        (gitignored)  input rasters and CSVs
├── docs/                   notes, TODO, plan, runbook, this file, glossary
└── manual_zone/            my area
    ├── reference/          old geocomp-2025 code, read-only, deleted once ported
    └── scratch/            learning and verification mini scripts
```

Files in `src/` that only contain a docstring are skeletons: the docstring
says what will go there.

## Inside `src/edf_dnn/`

| Folder or file | What it does |
|---|---|
| `data/grid.py` | Opens the rasters, selects bands by name, crops historical to the future grid, gives every pixel a `cell_id` and x/y |
| `data/blocks.py` | Spatial blocks and cross-validation folds (which pixels train, which validate) |
| `data/normalise.py` | Save and load the `MinMaxScaler` fitted on training pixels (`scaler.pkl` in each run) |
| `data/datasets.py` | Pixels to tensors on the training device (`pixel_dataset`) and mini-batches cut there (`batch_loader`); later one patch per sample for the CNN and U-Net |
| `models/mlp.py` | The MLPs from geocomp-2025 (single, two-layer, tapered) |
| `models/cnn.py`, `models/unet.py` | Patch CNN and U-Net |
| `train/loop.py` | `choose_device`, `train_one_epoch`, `predict` (per-pixel probabilities), `fit` (early stopping on validation ROC AUC) |
| `train/metrics.py` | One function for all metrics (ROC AUC, average precision, Brier and its skill score, log loss, recall, specificity, precision, TSS) |
| `train/run.py` | The one command that runs an experiment from its config file |
| `predict.py` | Applies a trained run to the future grid and writes a GeoTIFF |

Folds and scaling sit in `data/` because they decide what the model is
allowed to see.

## The other folders

- **`configs/`**: every setting that defines a run, so nothing is hard-coded
  in the scripts. Replaces the constants at the top of the old `train.py` and
  the one-folder-per-variant `run.py` files.
- **`runs/`**: replaces the old `model_out/` and `checkpoints/` folders. Each
  training run writes one folder with a copy of its config, the metrics, the
  trained weights and the scaler, so every result records what produced it.
- **`cluster_jobs/`**: PBS scripts, the small shell files submitted to the
  queue on NCAR Casper or Derecho (`qsub job.pbs`). Each one says, for
  example, "give me a GPU for 2 hours and run `run.py` with this config".
- **`reports/`**: Quarto `.qmd` files. They only read finished results from
  `runs/` and never train anything, so rendering is fast.
- **`tests/`**: small automated checks on tiny made-up data, run with
  `pixi run test`. If they pass, the code still does what it promised.
- **`benchmarks/`**: how long a training epoch takes on CPU and GPU, on
  synthetic data shaped like the real grid. `bench_training.py` runs it,
  `results.csv` holds the timings, `results_before_optimisation.csv` the
  timings before the data-path change of 2026-11-21, and `summarise.py` makes
  the README tables.
- **`manual_zone/`**: mine. `scratch/` holds scripts for stepping through code
  line by line and checking it on the real data; `reference/` holds the old
  code we port from.

## How a training run flows

```
configs/experiments/x.yaml
        │
        ▼
train/run.py ── data/grid.py ─────── rasters in data/
        │       data/blocks.py       which pixels are train / validation
        │       data/normalise.py    scaling from training pixels only
        │       data/datasets.py     pixels or patches as tensors
        │       models/...           the network
        │       train/loop.py        fit, validate, early stopping
        │       train/metrics.py     scores
        ▼
runs/x/  (config, metrics, weights, scaler)
        │
        ├──► reports/*.qmd          plots and tables
        └──► predict.py             future burn-probability GeoTIFF
```

## Old repo to new repo

| geocomp-2025 | EDF-DNN |
|---|---|
| `src/models.py` | `src/edf_dnn/models/mlp.py` |
| `src/data.py` | `src/edf_dnn/data/` |
| `src/train.py` | `src/edf_dnn/train/` |
| Constants in `train.py` + one `run.py` folder per variant | One YAML in `configs/experiments/` per variant |
| `mlp_*/model_out/`, `mlp_*/checkpoints/` | `runs/<experiment>/` |
| `mlp_*/*.qmd` | `reports/` |
| `cnn_2d/spatial_*` scripts | `src/edf_dnn/data/grid.py` |
| (none) | `cluster_jobs/` |
