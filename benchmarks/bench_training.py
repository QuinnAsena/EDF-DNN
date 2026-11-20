"""Benchmark the MLP training epoch on synthetic data shaped like the real grid.

Why synthetic: the input rasters are not in git (see configs/data.yaml), but
training speed depends only on the shape of the data, not its values. The
synthetic pixel table matches the development data: 563,475 complete pixels,
25 predictors (the provisional list in configs/data.yaml), 13.7% burned
(see docs/NOTES.md).

What is timed is exactly one epoch of `fit` in edf_dnn/train/loop.py, using
the project's own functions:
    train_one_epoch  over the training pixels (shuffled mini-batches),
    predict          over the validation pixels (in order, no gradients),
    compute_metrics  on those predictions,
with pixel_dataset / batch_loader building the loaders as run.py will.

One fold of a 5-fold cross-validation is mimicked: 80% of pixels for
training, 20% for validation. Two training sets, as in the experiment config:
    downsample : every burned training pixel plus as many unburned ones (50/50)
    full       : every training pixel, with pos_weight in the loss instead

Usage:
    pixi run python benchmarks/bench_training.py            # CPU, writes results
    pixi run -e cuda python benchmarks/bench_training.py --device cuda

Each setting runs one warm-up epoch (the first epoch pays one-off costs, such
as allocating memory), then the median of --repeats timed epochs is reported.
Results are appended to benchmarks/results.csv.
"""

import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from edf_dnn.data.datasets import batch_loader, pixel_dataset
from edf_dnn.models.mlp import FeedforwardTwo
from edf_dnn.train.loop import predict, train_one_epoch
from edf_dnn.train.metrics import compute_metrics

SEED = 1984                     # the project's fixed seed
PIXELS = 563_475                # complete pixels in the development data
VARIABLES = [f"var{k:02d}" for k in range(1, 26)]   # 25 predictors, as in configs/data.yaml
BURNED_SHARE = 0.137
RESULTS = Path(__file__).with_name("results.csv")


def synthetic_table(seed=SEED):
    """A pixel table with the real data's shape: cell_id, 25 predictors, burned01.

    The label depends weakly on a few predictors, so the validation metrics
    are real numbers (ROC AUC well above 0.5) rather than noise; the values
    themselves do not affect the timing.
    """
    rng = np.random.default_rng(seed)
    values = rng.random((PIXELS, len(VARIABLES)), dtype=np.float32)
    signal = values[:, :5].sum(1) + rng.normal(0, 1.0, PIXELS)
    cutoff = np.quantile(signal, 1 - BURNED_SHARE)       # top 13.7% "burned"
    table = pd.DataFrame(values, columns=VARIABLES)
    table.insert(0, "cell_id", np.arange(PIXELS))
    table["burned01"] = (signal > cutoff).astype(np.int8)
    return table


def split(table, seed=SEED):
    """One fold of a 5-fold cross-validation: 80% training, 20% validation."""
    rng = np.random.default_rng(seed)
    in_validation = rng.random(len(table)) < 0.2
    return table[~in_validation], table[in_validation]


def downsample(training, seed=SEED):
    """Every burned pixel plus an equal-sized random sample of unburned ones (50/50)."""
    burned = training[training.burned01 == 1]
    unburned = training[training.burned01 == 0].sample(len(burned), random_state=seed)
    return pd.concat([burned, unburned])


def time_epoch(model, training_loader, validation_loader, loss_function, optimizer, device):
    """One epoch of fit(), timed in its three parts (seconds)."""
    def sync():
        # GPU work runs asynchronously: wait for it to finish before reading the clock
        if device.type == "cuda":
            torch.cuda.synchronize()
    sync()
    start = time.perf_counter()
    train_one_epoch(model, training_loader, loss_function, optimizer, device)
    sync()
    trained = time.perf_counter()
    predictions = predict(model, validation_loader, device)
    sync()
    predicted = time.perf_counter()
    metrics = compute_metrics(predictions["label"], predictions["probability"])
    done = time.perf_counter()
    return {"train_s": trained - start, "predict_s": predicted - trained,
            "metrics_s": done - predicted, "epoch_s": done - start, "roc_auc": metrics["roc_auc"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 28],
                        help="CPU threads to try (torch.set_num_threads); ignored on cuda")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--batch", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--imbalance", nargs="+", default=["downsample", "full"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    table = synthetic_table()
    training_all, validation = split(table)
    scaler = MinMaxScaler().fit(training_all[VARIABLES])          # training pixels only
    training_sets = {"downsample": downsample(training_all), "full": training_all}
    validation_set = pixel_dataset(validation, VARIABLES, scaler)
    thread_settings = args.threads if device.type == "cpu" else [torch.get_num_threads()]
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""

    new_file = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as handle:
        writer = None
        for imbalance in args.imbalance:
            training = training_sets[imbalance]
            training_set = pixel_dataset(training, VARIABLES, scaler)
            burned = training.burned01.mean()
            # pos_weight = unburned / burned: each burned pixel counts as much as
            # all the unburned ones together, the alternative to downsampling
            pos_weight = None if imbalance == "downsample" else torch.tensor((1 - burned) / burned, device=device)
            for threads in thread_settings:
                if device.type == "cpu":
                    torch.set_num_threads(threads)
                for hidden in args.hidden:
                    for batch in args.batch:
                        torch.manual_seed(SEED)
                        model = FeedforwardTwo(len(VARIABLES), hidden, dropout=0.1).to(device)
                        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                        training_loader = batch_loader(training_set, batch, shuffle=True)
                        validation_loader = batch_loader(validation_set, batch, shuffle=False)
                        time_epoch(model, training_loader, validation_loader, loss_function, optimizer, device)  # warm-up
                        runs = [time_epoch(model, training_loader, validation_loader, loss_function, optimizer, device)
                                for _ in range(args.repeats)]
                        row = {"device": args.device, "gpu": gpu_name, "threads": threads if device.type == "cpu" else "",
                               "imbalance": imbalance, "training_pixels": len(training), "validation_pixels": len(validation),
                               "hidden": hidden, "batch": batch, "repeats": args.repeats}
                        for key in ("train_s", "predict_s", "metrics_s", "epoch_s"):
                            row[key] = round(statistics.median(r[key] for r in runs), 4)
                        row["train_pixels_per_s"] = round(len(training) / row["train_s"])
                        row["predict_pixels_per_s"] = round(len(validation) / row["predict_s"])
                        row["roc_auc"] = round(runs[-1]["roc_auc"], 4)
                        row["torch"] = torch.__version__
                        row["cpu"] = platform.processor() or platform.machine()
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(row))
                            if new_file:
                                writer.writeheader()
                        writer.writerow(row)
                        handle.flush()
                        print(f"{args.device:4} threads={row['threads']!s:>3} {imbalance:10} hidden={hidden:3} "
                              f"batch={batch:5}: epoch {row['epoch_s']:.3f} s (train {row['train_s']:.3f}, "
                              f"predict {row['predict_s']:.3f}, metrics {row['metrics_s']:.3f})", flush=True)


if __name__ == "__main__":
    main()
