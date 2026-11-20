# EDF-DNN

Deep neural networks (MLP, CNN, U-Net) for mapping burn risk on a 1 km grid,
trained on historical drivers and projected onto CMIP6 future climate.

## Setup (WSL / Linux)

```bash
curl -fsSL https://pixi.sh/install.sh | sh      # once per machine
cd ~/projects/EDF-DNN
pixi install                                    # builds the CPU environment from pixi.lock
pixi run test                                   # checks everything imports and runs
pixi run register-kernel                        # Jupyter kernel for Quarto reports
```

On a GPU node (e.g. NCAR Casper): `pixi install -e cuda`, then `pixi run -e cuda ...`.

Input data are not in git; see `configs/data.yaml` for the expected files in `data/`.

## Documentation

- [docs/repo-description.md](docs/repo-description.md): what every folder is for
- [docs/runbook.md](docs/runbook.md): setup and day-to-day commands
- [docs/TODO.md](docs/TODO.md) and [docs/NOTES.md](docs/NOTES.md): open work, and decisions so far
- [docs/glossary.md](docs/glossary.md): machine-learning terms in plain words

## Benchmarks

How long one training epoch takes, as `fit` in `src/edf_dnn/train/loop.py` runs
it: `train_one_epoch` on the training pixels, `predict` on the validation
pixels, then `compute_metrics`. The model is `FeedforwardTwo` (dropout 0.1,
Adam), at the hidden sizes and batch sizes of `configs/experiments/mlp_baseline.yaml`
and either side of them.

The input rasters are not in git, so the data are synthetic but shaped like the
development data: 563,475 complete pixels, the 25 predictors of
`configs/data.yaml`, 13.7% burned (see `docs/NOTES.md`). One fold of a 5-fold
cross-validation is mimicked (80% training, 20% validation), with the two
imbalance options: **downsampled** (all burned training pixels plus as many
unburned) and **full** (every training pixel, `pos_weight` in the loss). Each
number is the median of 3 epochs after one warm-up epoch.

**Machine:** Intel Xeon (Cascade Lake) VM, 28 vCPUs, 93 GB RAM; NVIDIA A100 80GB
PCIe. PyTorch 2.14.1 from `pixi.lock` (`default` environment on the CPU, `cuda`
environment with CUDA 12.9 on the GPU). Run on 2026-11-21 on a shared machine
with its other heavy jobs paused (load average about 5 at the start); the GPU was
otherwise idle, though an idle vLLM server held most of its memory.

**Downsampled training set** (123,494 training pixels, 112,624 validation pixels): seconds per epoch

| | hidden 128, batch 1,024 | hidden 128, batch 4,096 | hidden 128, batch 16,384 | hidden 256, batch 1,024 | hidden 256, batch 4,096 | hidden 256, batch 16,384 |
|---|---|---|---|---|---|---|
| CPU, 1 thread | 1.13 | 1.02 | 1.63 | 2.11 | 2.79 | 3.22 |
| CPU, 4 threads | 0.91 | 0.72 | 0.88 | 1.61 | 1.77 | 1.83 |
| CPU, 8 threads | 0.98 | 0.68 | 0.83 | 1.39 | 1.38 | 1.40 |
| CPU, 28 threads | 1.01 | 0.68 | 0.73 | 1.46 | 1.38 | 1.22 |
| GPU (A100) | 0.59 | 0.37 | 0.29 | 0.59 | 0.37 | **0.26** |

**Full training set** (450,851 training pixels, 112,624 validation pixels): seconds per epoch

| | hidden 128, batch 1,024 | hidden 128, batch 4,096 | hidden 128, batch 16,384 | hidden 256, batch 1,024 | hidden 256, batch 4,096 | hidden 256, batch 16,384 |
|---|---|---|---|---|---|---|
| CPU, 1 thread | 3.91 | 3.19 | 3.25 | 6.56 | 6.55 | 7.90 |
| CPU, 4 threads | 3.11 | 2.41 | 2.09 | 5.27 | 4.07 | 4.84 |
| CPU, 8 threads | 3.30 | 2.24 | 1.93 | 4.56 | 3.70 | 4.18 |
| CPU, 28 threads | 3.48 | 2.14 | 1.80 | 4.86 | 3.59 | 3.85 |
| GPU (A100) | 1.61 | 0.92 | 0.63 | 1.67 | 0.82 | **0.58** |

What this shows:

- **The baseline is about 0.7 s per epoch on a CPU** (downsampled, hidden 128,
  batch 4,096), in line with the "~1 s per epoch on the laptop CPU" noted in
  `mlp_baseline.yaml`.
- **More than 4–8 CPU threads barely helps.** The network is small (25 inputs, two
  hidden layers), so each batch is too little work to share out; 28 threads is no
  faster than 8.
- **The GPU is 1.9–3.8x faster than the best CPU setting at batch 4,096** on the
  downsampled set and 2.3–4.4x on the full set, and up to 4.8x and 6.6x at batch
  16,384. Its advantage grows with the batch size and the hidden width; at batch
  1,024 it gains little, because each step is too small to keep it busy.
- **About 0.17 s of every epoch is fixed:** predicting the 112,624 validation
  pixels (about 0.1 s) and computing the metrics with scikit-learn on the CPU
  (about 0.07 s). On the GPU at batch 16,384 that is over half the epoch.
- **The full training set costs about three times the downsampled one** per epoch,
  as it has 3.7 times the pixels.
- **For the whole grid search** in `mlp_baseline.yaml` (12 combinations, 5 folds),
  every 100 epochs per run costs about 1.7 h on 8 CPU threads and 37 min on the
  GPU with the downsampled set, or 5.0 h and 1.5 h with the full set (batch 4,096).

To rerun (results are appended to `benchmarks/results.csv`; the tables above
come from `benchmarks/summarise.py`):

```bash
pixi run python benchmarks/bench_training.py                    # CPU, 1/4/8/28 threads
pixi run -e cuda python benchmarks/bench_training.py --device cuda
pixi run python benchmarks/summarise.py
```

