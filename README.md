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
and either side of them. The pixel tables live on the training device
(`pixel_dataset(..., device=device)`), so batches are cut there.

The input rasters are not in git, so the data are synthetic but shaped like the
development data: 563,475 complete pixels, the 25 predictors of
`configs/data.yaml`, 13.7% burned (see `docs/NOTES.md`). One fold of a 5-fold
cross-validation is mimicked (80% training, 20% validation), with the two
imbalance options: **downsampled** (all burned training pixels plus as many
unburned) and **full** (every training pixel, `pos_weight` in the loss). Each
number is the median of 3 epochs (CPU) or 5 epochs (GPU) after one warm-up epoch.

**Machine:** Intel Xeon (Cascade Lake) VM, 28 vCPUs, 93 GB RAM; NVIDIA A100 80GB
PCIe. PyTorch 2.14.1 from `pixi.lock` (`default` environment on the CPU, `cuda`
environment with CUDA 12.9 on the GPU). Run on 2026-11-21 on a shared machine
with its other heavy jobs paused; the GPU was otherwise idle, though an idle
vLLM server held most of its memory. Repeat runs on this machine vary by up to
about 20%, more at batch 1,024.

**Downsampled training set** (123,494 training pixels, 112,624 validation pixels): seconds per epoch

| | hidden 128, batch 1,024 | hidden 128, batch 4,096 | hidden 128, batch 16,384 | hidden 256, batch 1,024 | hidden 256, batch 4,096 | hidden 256, batch 16,384 |
|---|---|---|---|---|---|---|
| CPU, 1 thread | 1.02 | 0.96 | 1.19 | 1.95 | 2.81 | 3.10 |
| CPU, 4 threads | 0.80 | 0.61 | 0.76 | 1.32 | 1.09 | 1.45 |
| CPU, 8 threads | 0.84 | 0.54 | 0.61 | 1.33 | 1.22 | 1.24 |
| CPU, 28 threads | 0.87 | 0.69 | 0.60 | 1.26 | 1.36 | 0.95 |
| GPU (A100) | 0.59 | 0.19 | **0.11** | 0.56 | 0.19 | 0.11 |

**Full training set** (450,851 training pixels, 112,624 validation pixels): seconds per epoch

| | hidden 128, batch 1,024 | hidden 128, batch 4,096 | hidden 128, batch 16,384 | hidden 256, batch 1,024 | hidden 256, batch 4,096 | hidden 256, batch 16,384 |
|---|---|---|---|---|---|---|
| CPU, 1 thread | 3.33 | 3.33 | 3.49 | 7.64 | 7.36 | 8.55 |
| CPU, 4 threads | 2.65 | 2.04 | 1.82 | 4.82 | 3.78 | 4.07 |
| CPU, 8 threads | 2.68 | 1.67 | 1.53 | 4.45 | 3.27 | 3.82 |
| CPU, 28 threads | 3.10 | 1.87 | 1.76 | 5.25 | 3.39 | 3.45 |
| GPU (A100) | 2.11 | 0.39 | 0.19 | 1.91 | 0.56 | **0.17** |

What this shows:

- **The baseline is about 0.5–0.6 s per epoch on 4–8 CPU threads and 0.19 s on
  the GPU** (downsampled, hidden 128, batch 4,096).
- **More than 4–8 CPU threads barely helps.** The network is small (25 inputs, two
  hidden layers), so each batch is too little work to share out; 28 threads is no
  faster than 8.
- **The GPU is 2.8–5.9x faster than the best CPU setting at batch 4,096,** and
  about 5–9x (downsampled) and 9–20x (full) at batch 16,384.
- **At batch 1,024 the GPU gains little.** Every training step, whatever its size,
  costs the CPU about 3 ms on this machine to send its ~50 small operations to the
  GPU, and at batch 1,024 an epoch has 121 (downsampled) or 441 (full) steps.
  Larger batches mean fewer steps.
- **About 0.08 s of every epoch is the metrics,** computed with scikit-learn on
  the CPU; on the GPU at batch 16,384 that is most of the epoch.
- **For the whole grid search** in `mlp_baseline.yaml` (12 combinations, 5 folds),
  every 100 epochs per run costs about 1.5 h on 8 CPU threads and 19 min on the
  GPU with the downsampled set, or 4.1 h and 48 min with the full set (batch 4,096).

### The data-path optimisation (2026-11-21)

The first benchmark showed that on the GPU most of an epoch went on fetching
batches, not training: torch's `DataLoader` gathered each batch's rows from the
table in the CPU's memory, and every batch was then copied to the GPU. Now
`pixel_dataset(..., device=device)` puts the whole pixel table on the training
device once (about 60 MB), and `batch_loader` cuts each batch there, with one
indexing step per tensor. `train_one_epoch` also adds up the loss on the device
and reads it once per epoch instead of after every batch, and `predict` copies
the predictions back once instead of per batch. See `docs/NOTES.md` (2026-11-21).

**Before and after** (batch 4,096; before = `benchmarks/results_before_optimisation.csv`): seconds per epoch

| | downsampled, hidden 128 | downsampled, hidden 256 | full, hidden 128 | full, hidden 256 |
|---|---|---|---|---|
| CPU, 1 thread | 1.02 → 0.96 (1.1x) | 2.79 → 2.81 (1.0x) | 3.19 → 3.33 (1.0x) | 6.55 → 7.36 (0.9x) |
| CPU, 4 threads | 0.72 → 0.61 (1.2x) | 1.77 → 1.09 (1.6x) | 2.41 → 2.04 (1.2x) | 4.07 → 3.78 (1.1x) |
| CPU, 8 threads | 0.68 → 0.54 (1.3x) | 1.38 → 1.22 (1.1x) | 2.24 → 1.67 (1.3x) | 3.70 → 3.27 (1.1x) |
| CPU, 28 threads | 0.68 → 0.69 (1.0x) | 1.38 → 1.36 (1.0x) | 2.14 → 1.87 (1.1x) | 3.59 → 3.39 (1.1x) |
| GPU (A100) | 0.37 → 0.19 (1.9x) | 0.37 → 0.19 (1.9x) | 0.92 → 0.39 (2.4x) | 0.82 → 0.56 (1.5x) |

- **On the GPU an epoch is 1.5–2.4x faster at batch 4,096 and 2.4–3.4x faster at
  16,384.** Alternating runs of the old and new code confirmed the gain at 4,096
  (training 0.40–0.46 s against 0.62–0.92 s).
- **On the CPU it is up to 1.3x faster** (most settings), from skipping the
  `DataLoader`'s bookkeeping for every batch. The 1-thread differences are within
  run-to-run noise.
- **At batch 1,024 there is no reliable change.** Each step is limited by the CPU
  sending work to the GPU, and on this shared machine that speed varied by up to
  2x from minute to minute. Tests that alternated old and new within one process
  had the new code as fast or faster; alternating whole benchmark runs had it
  slower (training 1.6–2.0 s against 1.3–2.0 s for the full set).

To rerun (results are appended to `benchmarks/results.csv`; the tables above
come from `benchmarks/summarise.py`):

```bash
pixi run python benchmarks/bench_training.py                    # CPU, 1/4/8/28 threads
pixi run -e cuda python benchmarks/bench_training.py --device cuda --repeats 5
pixi run python benchmarks/summarise.py
```
