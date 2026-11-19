"""Reprex: how pixel_dataset and batch_loader shuffle pixels and split them into mini-batches.

Ten made-up pixels, batches of 4. Run it line by line with Shift+Enter, or all
at once with:  pixi run python manual_zone/scratch/reprex_batches.py

The same steps in R, for comparison:
    order  <- sample(nrow(table))                          # RandomSampler
    chunks <- split(order, ceiling(seq_along(order) / 4))  # BatchSampler
    for (rows in chunks) batch <- table[rows, ]            # DataLoader
"""

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from edf_dnn.data.datasets import batch_loader, pixel_dataset

# ---------------------------------------------------------------------------
# 1. Ten pixels, built so every row can be checked by eye:
#    elevation = 10 x cell_id, and the even cell_ids burned
# ---------------------------------------------------------------------------
table = pd.DataFrame({"cell_id": range(101, 111)})
table["elevation"] = table["cell_id"] * 10.0
table["burned01"] = (table["cell_id"] % 2 == 0).astype(float)
print(table)

# ---------------------------------------------------------------------------
# 2. Scale and convert to tensors
# ---------------------------------------------------------------------------
scaler = MinMaxScaler().fit(table[["elevation"]])
dataset = pixel_dataset(table, ["elevation"], scaler)
print(len(dataset))        # 10 pixels
print(dataset[0])          # pixel 0: (scaled elevation, label, cell_id) = (0, 0, 101)
print(dataset[9])          # pixel 9: (1, 1, 110)

# ---------------------------------------------------------------------------
# 3. Training loader: shuffled batches of 4 -> 4 + 4 + 2 pixels per epoch,
#    in a new order every epoch
# ---------------------------------------------------------------------------
torch.manual_seed(1984)    # same shuffles every time the script runs
training_loader = batch_loader(dataset, batch_size=4, shuffle=True)

for epoch in [1, 2]:
    print(f"epoch {epoch}")
    for batch_features, batch_labels, batch_cell_ids in training_loader:
        # Undo the scaling to see the original elevations: each must be
        # 10 x its cell_id, and each even cell_id must have label 1
        elevations = scaler.inverse_transform(batch_features.numpy()).round().ravel()
        print("  cell_id", batch_cell_ids.tolist(),
              " elevation", elevations.tolist(),
              " burned", batch_labels.tolist())

# ---------------------------------------------------------------------------
# 4. Check: in one epoch, every pixel appears exactly once
# ---------------------------------------------------------------------------
seen = [cell_id for _, _, batch_cell_ids in training_loader for cell_id in batch_cell_ids.tolist()]
print(len(seen), sorted(seen) == list(table["cell_id"]))   # 10 True

# ---------------------------------------------------------------------------
# 5. Evaluation loader: no shuffling, so batches follow the table's order
# ---------------------------------------------------------------------------
evaluation_loader = batch_loader(dataset, batch_size=4, shuffle=False)
for _, _, batch_cell_ids in evaluation_loader:
    print(batch_cell_ids.tolist())
