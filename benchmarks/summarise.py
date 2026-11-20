"""Turn benchmarks/results.csv into the Markdown tables shown in the README.

Usage:
    pixi run python benchmarks/summarise.py

Prints one table per training set (downsampled, full): median seconds per
epoch for every device / thread count, model width and batch size, then the
GPU speed-up over the best CPU setting, then (if
benchmarks/results_before_optimisation.csv exists) the epoch times before and
after the data-path optimisation, at the experiment config's batch size.
"""

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).with_name("results.csv")
BEFORE = Path(__file__).with_name("results_before_optimisation.csv")


def device_label(row):
    """'CPU, 8 threads' or 'GPU (A100)': one row label per hardware setting."""
    if row["device"] == "cuda":
        return f"GPU ({row['gpu'].replace('NVIDIA ', '').split(' ')[0]})"
    return f"CPU, {int(row['threads'])} thread{'s' if int(row['threads']) > 1 else ''}"


def main():
    results = pd.read_csv(RESULTS)
    results["setting"] = results.apply(device_label, axis=1)
    order = list(dict.fromkeys(results.sort_values(["device", "threads"]).setting))
    for imbalance, label in (("downsample", "Downsampled training set"), ("full", "Full training set")):
        part = results[results.imbalance == imbalance]
        pixels = int(part.training_pixels.iloc[0])
        table = part.pivot_table(index="setting", columns=["hidden", "batch"], values="epoch_s").reindex(order)
        print(f"**{label}** ({pixels:,} training pixels, {int(part.validation_pixels.iloc[0]):,} validation pixels): "
              "seconds per epoch\n")
        header = "| | " + " | ".join(f"hidden {h}, batch {b:,}" for h, b in table.columns) + " |"
        print(header)
        print("|---" * (len(table.columns) + 1) + "|")
        for setting, row in table.iterrows():
            best = row.min()
            cells = [f"**{v:.2f}**" if v == best and setting.startswith("GPU") else f"{v:.2f}" for v in row]
            print(f"| {setting} | " + " | ".join(cells) + " |")
        cpu_best = part[part.device == "cpu"].groupby(["hidden", "batch"]).epoch_s.min()
        gpu = part[part.device == "cuda"].set_index(["hidden", "batch"]).epoch_s
        speedup = (cpu_best / gpu).dropna()
        print(f"\nGPU speed-up over the best CPU thread count: {speedup.min():.1f}x to {speedup.max():.1f}x "
              f"(batch 4,096: {speedup.xs(4096, level='batch').min():.1f}x to {speedup.xs(4096, level='batch').max():.1f}x)\n")


def before_after(batch=4096):
    """Seconds per epoch before and after the optimisation, at one batch size."""
    keys = ["setting", "imbalance", "hidden"]
    tables = []
    for path in (BEFORE, RESULTS):
        frame = pd.read_csv(path)
        frame["setting"] = frame.apply(device_label, axis=1)
        tables.append(frame[frame.batch == batch].set_index(keys).epoch_s)
    both = pd.concat(tables, axis=1, keys=["before", "after"]).dropna()
    order = list(dict.fromkeys(pd.read_csv(RESULTS).sort_values(["device", "threads"])
                               .apply(device_label, axis=1)))
    print(f"**Before and after the data-path optimisation** (batch {batch:,}): seconds per epoch\n")
    print("| | downsampled, hidden 128 | downsampled, hidden 256 | full, hidden 128 | full, hidden 256 |")
    print("|---|---|---|---|---|")
    for setting in order:
        cells = []
        for imbalance in ("downsample", "full"):
            for hidden in (128, 256):
                b, a = both.loc[(setting, imbalance, hidden)]
                cells.append(f"{b:.2f} → {a:.2f} ({b / a:.1f}x)")
        print(f"| {setting} | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    main()
    if BEFORE.exists():
        before_after()
