"""The training loop: train on mini-batches, check on validation pixels, stop early.

Contents
--------
- choose_device   : the GPU if there is one, else the CPU.
- train_one_epoch : one pass over the training batches, updating the weights after each.
- predict         : burn probability for every pixel in a loader, with its label and cell_id.
- fit             : train_one_epoch + predict, epoch after epoch, until validation
                    ROC AUC stops improving; the model keeps its best epoch's weights.

Early stopping watches validation ROC AUC, not validation loss: with
downsampled training, the validation loss wobbles up and down while the
ranking keeps improving (see NOTES.md, 2026-11-19).
"""

import copy

import pandas as pd
import torch

from edf_dnn.train.metrics import compute_metrics


def choose_device(setting="auto"):
    """Where the model and data live: "auto" picks the GPU if there is one, else the CPU.

    "cpu" or "cuda" force one. Everything else just moves tensors to this
    device, so the same code runs on the laptop (CPU) and on Casper (GPU).
    """
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


def train_one_epoch(model, loader, loss_function, optimizer, device):
    """One pass over all training batches, updating the weights after each batch.

    Returns the mean training loss over the epoch's pixels.
    """
    model.train()   # training mode: dropout on
    # The running total stays on the device: loss.item() on every batch would
    # make the CPU wait for the GPU each time, stalling it between batches.
    # Reading the total once, at the end, costs one wait per epoch.
    total_loss = torch.zeros((), device=device)
    pixel_count = 0
    for features, labels, _ in loader:   # cell_ids are not needed for training
        # .to(device) is free when the dataset already lives there (pixel_dataset(device=...))
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()                     # clear the last batch's gradients
        logits = model(features).squeeze(1)       # one logit per pixel: (batch, 1) -> (batch,)
        loss = loss_function(logits, labels)      # how wrong, on average, on this batch
        loss.backward()                           # gradients: which way to move each weight
        optimizer.step()                          # move every weight a small step that way
        # Weight each batch's loss by its size (the last batch is smaller);
        # detach() keeps the total out of the gradient bookkeeping
        total_loss += loss.detach() * len(labels)
        pixel_count += len(labels)
    return total_loss.item() / pixel_count


def predict(model, loader, device):
    """Predicted burn probability for every pixel in a loader.

    Returns a DataFrame with cell_id, label and probability, one row per
    pixel. This is what gets saved per fold, and matched back by cell_id.
    """
    model.eval()   # evaluation mode: dropout off, so predictions are repeatable
    all_logits, all_labels, all_cell_ids = [], [], []
    with torch.no_grad():   # no gradients needed when only predicting: faster, less memory
        for features, labels, cell_ids in loader:
            all_logits.append(model(features.to(device)).squeeze(1))
            all_labels.append(labels)
            all_cell_ids.append(cell_ids)
    # Stack the batches (like do.call(rbind, ...)) and bring them back to the CPU
    # once, rather than building a small table per batch: on a GPU each batch's
    # copy back would make the CPU wait for the GPU.
    return pd.DataFrame({
        "cell_id": torch.cat(all_cell_ids).cpu().numpy(),
        "label": torch.cat(all_labels).cpu().numpy(),
        # sigmoid turns logits into probabilities, like plogis() in R
        "probability": torch.sigmoid(torch.cat(all_logits)).cpu().numpy(),
    })


def fit(model, training_loader, validation_loader, loss_function, optimizer, device,
        max_epochs, patience):
    """Train until validation ROC AUC has not improved for `patience` epochs.

    Afterwards the model holds the weights of its best epoch, not its last.

    Returns
    -------
    history    : DataFrame, one row per epoch: epoch, training_loss, and every
                 metric from compute_metrics on the validation pixels, named
                 validation_roc_auc, validation_log_loss, ...
    best_epoch : the epoch whose weights the model now holds
    """
    history = []
    best_roc_auc, best_epoch, best_weights = -1.0, 0, None

    for epoch in range(1, max_epochs + 1):
        training_loss = train_one_epoch(model, training_loader, loss_function, optimizer, device)
        predictions = predict(model, validation_loader, device)
        metrics = compute_metrics(predictions["label"], predictions["probability"])

        # One row of the history: the epoch, the training loss, then each
        # validation metric with "validation_" in front of its name
        row = {"epoch": epoch, "training_loss": training_loss}
        for name, value in metrics.items():
            row[f"validation_{name}"] = value
        history.append(row)

        if metrics["roc_auc"] > best_roc_auc:
            # New best: keep a copy of these weights. deepcopy freezes them;
            # state_dict() on its own would keep changing as training goes on.
            best_roc_auc, best_epoch = metrics["roc_auc"], epoch
            best_weights = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break   # no improvement for `patience` epochs: stop

    # Go back to the best epoch's weights
    model.load_state_dict(best_weights)
    return pd.DataFrame(history), best_epoch
