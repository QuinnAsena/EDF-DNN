"""The whole MLP pipeline by hand, top to bottom: one fold, a few epochs, no bookkeeping.

This is the plain-script version of what run.py does. run.py (steps b and c)
wraps these same lines in loops, over hyperparameter settings and folds, and
saves the results to a run folder.

Run it line by line with Shift+Enter (console starts in the repo root), or all
at once (about a minute) with:  pixi run python manual_zone/scratch/walkthrough_mlp.py
(as a whole script the plots are skipped unless you add plt.show()).

Only 20 epochs: enough to see every step work, far too few for real results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import rioxarray
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

from edf_dnn.data.datasets import batch_loader, pixel_dataset
from edf_dnn.data.grid import crop_to_grid, open_stack, to_pixel_table
from edf_dnn.data.splits import assign_folds, downsample, hold_out_test
from edf_dnn.models.mlp import FeedforwardTwo
from edf_dnn.train.loop import choose_device, fit, predict_probabilities
from edf_dnn.train.metrics import compute_metrics

# ---------------------------------------------------------------------------
# 0. Settings: in run.py these come from configs/data.yaml and the experiment config
# ---------------------------------------------------------------------------
config = yaml.safe_load(open("configs/data.yaml"))
target, variables, seed = config["target"], config["variables"], config["random_seed"]
hidden_size, dropout, learning_rate = 256, 0.2, 1e-3   # geocomp-2025's best two-layer setting
batch_size, max_epochs, patience = 4096, 20, 30        # 20 epochs: a quick look only
device = choose_device("auto")
print(device)                                          # cpu on the laptop

# ---------------------------------------------------------------------------
# 1. Load: raster -> crop to the future grid -> one row per pixel
# ---------------------------------------------------------------------------
stack = open_stack(f"{config['data_dir']}/{config['historical_raster']}", [target] + variables)
template = rioxarray.open_rasterio(f"{config['data_dir']}/{config['future_raster']}")
historical_data = to_pixel_table(crop_to_grid(stack, template))
print(historical_data.shape)                           # (563475, 29): cell_id, x, y, target, 25 predictors

# ---------------------------------------------------------------------------
# 2. Split: hold out the test set, give the rest folds, take fold 1 to validate
# ---------------------------------------------------------------------------
pool, test = hold_out_test(historical_data, target, config["test_fraction"], seed)
pool["fold"] = assign_folds(pool, target, config["n_folds"], seed)
training, validation = pool[pool["fold"] != 1], pool[pool["fold"] == 1]
print(len(training), len(validation), len(test))       # 405,701 / 101,426 / 56,348 (test unused here)

# ---------------------------------------------------------------------------
# 3. Scale (fitted on the training pixels only), then downsample them to 50/50
# ---------------------------------------------------------------------------
scaler = MinMaxScaler().fit(training[variables])
balanced = downsample(training, target, seed)
print(len(balanced), balanced[target].mean())          # 110,866 pixels, 0.5 burned

# ---------------------------------------------------------------------------
# 4. Tensors and mini-batches
# ---------------------------------------------------------------------------
training_loader = batch_loader(pixel_dataset(balanced, variables, scaler, target),
                               batch_size, shuffle=True)
validation_loader = batch_loader(pixel_dataset(validation, variables, scaler, target),
                                 batch_size, shuffle=False)
features, labels, cell_ids = next(iter(training_loader))   # peek at the first batch
print(features.shape, labels.shape, cell_ids[:5])      # (4096, 25), (4096,), five cell_ids

# ---------------------------------------------------------------------------
# 5. Model, optimizer and loss function
# ---------------------------------------------------------------------------
torch.manual_seed(seed)                                # same starting weights and shuffles every run
model = FeedforwardTwo(len(variables), hidden_size, dropout).to(device)
print(model)                                           # the layers and their sizes
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # moves the weights
loss_function = torch.nn.BCEWithLogitsLoss()           # scores how wrong the predictions are

# ---------------------------------------------------------------------------
# 6. Train, checking the validation pixels after every epoch (early stopping)
# ---------------------------------------------------------------------------
history, best_epoch = fit(model, training_loader, validation_loader, loss_function,
                          optimizer, device, max_epochs, patience)
print(history[["epoch", "training_loss", "validation_roc_auc",
               "validation_average_precision"]].round(3).to_string(index=False))
print(best_epoch)                                      # the model now holds this epoch's weights

# ---------------------------------------------------------------------------
# 7. Predict the validation pixels and score them
# ---------------------------------------------------------------------------
predictions = predict_probabilities(model, validation_loader, device)
print(predictions.head())                              # cell_id, label, probability
metrics = compute_metrics(predictions["label"], predictions["probability"])
print(pd.Series(metrics).round(3))                     # ROC AUC ~0.78 after only 20 epochs
# The Brier skill score comes out NEGATIVE (about -0.57): trained on 50/50
# data, the model predicts probabilities far above the real 13.7% burn rate,
# so as probabilities they score worse than always saying 13.7%. The ranking
# (ROC AUC, average precision) is fine; the probabilities need correcting for
# the sampling ratio before mapping absolute burn probability (TODO.md).

# Predictions carry cell_id, so x/y (or anything else) joins back by cell_id
mapped = predictions.merge(validation[["cell_id", "x", "y"]], on="cell_id")
print(mapped.head())

# ---------------------------------------------------------------------------
# 8. Look at it
# ---------------------------------------------------------------------------
figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
# Learning curve: training loss falls, validation ROC AUC rises
left.plot(history["epoch"], history["training_loss"], label="training loss")
left.plot(history["epoch"], history["validation_roc_auc"], label="validation ROC AUC")
left.set_xlabel("epoch")
left.legend()
# Predicted probabilities for burned and unburned validation pixels: the
# better the model, the less the two histograms overlap. Both sit high,
# because a model trained on 50/50 data overestimates the real 13.7%.
for label, name in [(0, "unburned"), (1, "burned")]:
    right.hist(predictions.loc[predictions["label"] == label, "probability"],
               bins=50, alpha=0.5, density=True, label=name)
right.set_xlabel("predicted burn probability")
right.legend()

# Tidy up before exit, which avoids rioxarray's harmless shutdown message
del stack, template
import gc
gc.collect()
