# CLAUDE.md

Durable reference for this repo. For "what's been tried and why" see
[docs/NOTES.md](docs/NOTES.md); for open work see [docs/TODO.md](docs/TODO.md).
Keep this file free of status updates: it describes what is stable.

## Project

Burn-risk mapping with deep neural networks for an EDF project. Binary target
`burned01` on a 1 km grid. Models are trained on historical drivers and
applied to CMIP6 future-climate drivers to project burn probability.

Scope: 2D spatial data only. Models: MLP (per pixel), patch CNN, U-Net.
Time-series models (GRU, CNN-GRU) are benched: no annual data exist.

Successor to the geocomp-2025 repo (archived), which holds the original MLP
experiments and reference code.

## Environment

- [pixi](https://pixi.sh), conda-forge, `linux-64`, Python 3.13.
- Repo lives in the WSL Linux filesystem (`~/projects/EDF-DNN`), not `/mnt/c`:
  the Windows drive is roughly 10x slower to read from WSL.
- Two environments: `default` (CPU PyTorch, local) and `cuda` (GPU PyTorch, for
  NCAR Casper/Derecho; install there with `pixi install -e cuda`).
- The package `src/edf_dnn` is installed in editable mode, so `import edf_dnn`
  works from anywhere. Never use `sys.path.insert`.
- `pixi run test` runs pytest. `pixi run register-kernel` registers the
  "EDF-DNN" Jupyter kernel for Quarto; re-run after a Python version change.

## Layout

| Path | Purpose |
|---|---|
| `src/edf_dnn/data/` | raster reading by band name, spatial folds, Datasets, scaling |
| `src/edf_dnn/models/` | architectures; all return raw logits |
| `src/edf_dnn/train/` | training loop, metrics, `run.py` entry point |
| `src/edf_dnn/predict.py` | trained run -> future burn-probability GeoTIFF |
| `configs/` | `data.yaml` (shared) and one YAML per experiment |
| `tests/` | pytest on tiny synthetic data |
| `runs/` | gitignored; one folder per experiment with config, metrics, checkpoints |
| `cluster_jobs/` | PBS scripts for Casper/Derecho |
| `reports/` | Quarto output; reads only from `runs/` |
| `data/` | gitignored; input data (development copies until the master-grid data arrive) |
| `docs/` | NOTES, TODO, plan snapshot, runbook, repo description, ML glossary |
| `manual_zone/` | the user's area: `reference/` (old geocomp-2025 code, read-only, for porting) and `scratch/` (learning and verification scripts) |

## Conventions

- Select raster bands by name, never by position.
- Every pixel carries `cell_id` and x/y; never link tables by row order.
- Scaling statistics and class balancing come from training folds only.
- Use mini-batches via DataLoader and move tensors to `device`, so code runs
  unchanged on CPU or GPU.
- Readable variable names (`hidden_size`, not `hid`), and comments that
  explain the why, including a sentence on neural-network idioms.
- Quarto `.qmd` for reports; no Jupyter notebooks.
- Fixed random seed `1984`.
- When a new ML term enters the code, add it to [docs/glossary.md](docs/glossary.md).
- Folder or file moves are reflected in [docs/repo-description.md](docs/repo-description.md).
