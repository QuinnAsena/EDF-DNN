"""Do the training numbers match what theory predicts? Four checks on the real data.

loop.py is hard to step through line by line, so instead each check here has a
known expected answer (after Karpathy 2019, "A Recipe for Training Neural
Networks", http://karpathy.github.io/2019/04/25/recipe/):

1. Loss before any training: about 0.693, which is -log(0.5)
2. Memorise 64 pixels: the training loss falls to about 0
3. Baseline: logistic regression on the same fold, the number to beat
4. Early stopping on the real data: best epoch, stop epoch, restored weights

Run it line by line with Shift+Enter (console starts in the repo root), or all
at once (about 3 minutes) with:  pixi run python manual_zone/scratch/verify_training.py
(as a whole script the plot is skipped unless you add plt.show()).
"""

import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from edf_dnn.data.batching import batch_loader, table_to_tensors
from edf_dnn.data.grid import crop_to_grid, open_stack, to_pixel_table
from edf_dnn.data.splits import assign_folds, downsample, hold_out_test
from edf_dnn.models.mlp import FeedforwardTwo
from edf_dnn.train.loop import choose_device, fit, predict_probabilities, train_one_epoch
from edf_dnn.train.metrics import compute_metrics

# ---------------------------------------------------------------------------
# 0. Fold 1, scaled and downsampled, as run.py's train_one_fold does it
# ---------------------------------------------------------------------------
config = yaml.safe_load(open("configs/data.yaml"))
target, variables, seed = config["target"], config["variables"], config["random_seed"]
device = choose_device("auto")

stack = open_stack(f"{config['data_dir']}/{config['historical_raster']}", [target] + variables)
template = rioxarray.open_rasterio(f"{config['data_dir']}/{config['future_raster']}")
historical_data = to_pixel_table(crop_to_grid(stack, template))
pool, test = hold_out_test(historical_data, target, config["test_fraction"], seed)
pool["fold"] = assign_folds(pool, target, config["n_folds"], seed)
training, validation = pool[pool["fold"] != 1], pool[pool["fold"] == 1]

scaler = MinMaxScaler().fit(training[variables])
balanced = downsample(training, target, seed)
training_dataset = table_to_tensors(balanced, variables, scaler, target)
validation_loader = batch_loader(table_to_tensors(validation, variables, scaler, target),
                                 4096, shuffle=False)
loss_function = torch.nn.BCEWithLogitsLoss()

# ---------------------------------------------------------------------------
# 1. Loss before any training: about 0.693
# ---------------------------------------------------------------------------
# A fresh network starts with small random weights, so its logits sit near 0
# and its probabilities near 0.5 (plogis(0) = 0.5 in R). Saying 0.5 for every
# pixel costs -log(0.5) = 0.693 per pixel, whatever the labels. A value far
# from that would mean the wrong loss, wrong labels or unscaled inputs.
torch.manual_seed(seed)
model = FeedforwardTwo(len(variables), hidden_size=256, dropout=0.2).to(device)
features, labels, _ = training_dataset.tensors       # all balanced training pixels at once
model.eval()                                          # dropout off: just look, don't train
with torch.no_grad():
    logits = model(features.to(device)).squeeze(1)
print(-np.log(0.5))                                   # 0.693: the expected value
print(loss_function(logits, labels.to(device)).item())   # 0.695: close to 0.693
print(torch.sigmoid(logits).mean().item())            # 0.48: close to 0.5

# ---------------------------------------------------------------------------
# 2. Memorise 64 pixels: the training loss should fall to about 0
# ---------------------------------------------------------------------------
# 64 pixels against ~73,000 weights: the network can learn each pixel by
# heart. If the loss doesn't fall, the gradients and the optimizer aren't
# connected (e.g. a missing optimizer.step()). Dropout off, so nothing holds
# it back. This uses the real train_one_epoch.
tiny = balanced.groupby(target).head(32)              # 32 burned + 32 unburned pixels
tiny_loader = batch_loader(table_to_tensors(tiny, variables, scaler, target), 64, shuffle=True)
torch.manual_seed(seed)
model = FeedforwardTwo(len(variables), hidden_size=256, dropout=0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(1, 501):
    tiny_loss = train_one_epoch(model, tiny_loader, loss_function, optimizer, device)
    if epoch in (1, 10, 100, 500):
        print(epoch, round(tiny_loss, 4))             # 0.70, 0.56, 0.02, 0.0001
tiny_predictions = predict_probabilities(model, tiny_loader, device)
print(compute_metrics(tiny_predictions["label"], tiny_predictions["probability"])["roc_auc"])   # 1.0

# ---------------------------------------------------------------------------
# 3. Baseline: logistic regression on the same pixels, the number to beat
# ---------------------------------------------------------------------------
# Logistic regression is a network with no hidden layer: one weighted sum of
# the 25 predictors, then plogis(). Like glm(burned01 ~ ., family = binomial)
# in R; C=np.inf means no penalty, as in glm. Same scaled and downsampled
# training pixels, same validation pixels. The MLP's hidden layers should beat it.
baseline = LogisticRegression(C=np.inf, max_iter=1000)
baseline.fit(scaler.transform(balanced[variables]), balanced[target])
# predict_proba gives one column per class; the second is P(burned)
baseline_probabilities = baseline.predict_proba(scaler.transform(validation[variables]))[:, 1]
baseline_metrics = compute_metrics(validation[target], baseline_probabilities)
print(round(baseline_metrics["roc_auc"], 3))          # 0.712

# ---------------------------------------------------------------------------
# 4. Early stopping on the real data, with a short patience so it triggers
# ---------------------------------------------------------------------------
# dev_quick never stops early (20 epochs, patience 30). Even patience 5 never
# triggered: on 2026-11-24 this same model was still improving at epoch 200
# (validation ROC AUC 0.874, above geocomp-2025's 0.861), which is why the
# real configs use patience 30 and up to 1000 epochs. Patience 2 here, with
# everything else as in the real runs, so the stopping rule gets exercised:
patience = 2
torch.manual_seed(seed)
model = FeedforwardTwo(len(variables), hidden_size=256, dropout=0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
training_loader = batch_loader(training_dataset, 4096, shuffle=True)
history, best_epoch = fit(model, training_loader, validation_loader, loss_function,
                          optimizer, device, max_epochs=200, patience=patience)

# (a) The best epoch is the one with the highest validation ROC AUC in the history
print(best_epoch, history.loc[history["validation_roc_auc"].idxmax(), "epoch"])   # 84 84
# (b) Training stopped exactly `patience` epochs after it
print(len(history), best_epoch + patience)            # 86 86
# (c) The model went back to the best epoch's weights: predicting again
#     reproduces that epoch's AUC exactly, not the last epoch's
predictions = predict_probabilities(model, validation_loader, device)
restored_auc = compute_metrics(predictions["label"], predictions["probability"])["roc_auc"]
best_epoch_auc = history.loc[history["epoch"] == best_epoch, "validation_roc_auc"].item()
last_epoch_auc = history["validation_roc_auc"].iloc[-1]
print(restored_auc == best_epoch_auc, round(last_epoch_auc, 4), round(best_epoch_auc, 4))   # True 0.8406 0.8415
# (d) Against the baseline
print(round(best_epoch_auc, 3), "MLP vs", round(baseline_metrics["roc_auc"], 3), "logistic regression")   # 0.842 vs 0.712

# The validation curve, the best epoch and the baseline in one picture
figure, axis = plt.subplots(figsize=(7, 4))
axis.plot(history["epoch"], history["validation_roc_auc"], label="MLP, validation ROC AUC")
axis.axhline(baseline_metrics["roc_auc"], color="grey", linestyle="--", label="logistic regression")
axis.axvline(best_epoch, color="black", linestyle=":", label=f"best epoch ({best_epoch})")
axis.set_xlabel("epoch")
axis.legend()

# Tidy up before exit, which avoids rioxarray's harmless shutdown message
del stack, template
import gc
gc.collect()
