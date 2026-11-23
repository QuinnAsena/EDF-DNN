"""How pixels reach the network: PyTorch Datasets and mini-batch loaders.

Contents
--------
- pixel_dataset : pixel table -> TensorDataset of (features, labels, cell_ids),
                  one sample per pixel (the MLP's view; replaces the old to_tensors).
- batch_loader  : hands a dataset out in mini-batches, shuffled for training
                  or in order for evaluation.

A Dataset answers two questions: how many samples are there, and what is
sample number i? A DataLoader then deals the samples out in mini-batches.
Training on mini-batches instead of all pixels at once (full-batch, as in
geocomp-2025) gives many small weight updates per epoch, and is what lets the
CNN fit in memory later.

Planned (phase 4)
-----------------
- PatchDataset : one sample = a square window of pixels around a location
                 (the CNN and U-Net view). A hand-written Dataset, because
                 patches are cut out on the fly. To work with batch_loader,
                 its __getitem__ must accept a list of row numbers and return
                 the whole batch (batch_loader asks for dataset[rows]).
"""

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)


def pixel_dataset(table, variables, scaler, target="burned01"):
    """Turn a pixel table into a PyTorch dataset with one sample per pixel.

    Parameters
    ----------
    table     : pixel table (rows of to_pixel_table), already split: e.g. the
                training pixels of one fold, or its validation pixels
    variables : names of the predictor columns, as the scaler was fitted on
    scaler    : MinMaxScaler fitted on the TRAINING pixels (see normalise.py)
    target    : name of the label column

    Returns
    -------
    A TensorDataset: dataset[i] is (features, label, cell_id) for row i of the
    table, and len(dataset) is the number of pixels. Every pixel is kept,
    burned and unburned; nothing is filtered.
    """
    # Predictors: scaled to 0-1 (sklearn checks the names match the fit), as
    # float32, the precision networks compute in. Shape (pixels, variables).
    features = torch.tensor(scaler.transform(table[variables]), dtype=torch.float32)

    # Labels: the whole 0/1 column, like table$burned01 in R. float32 because
    # the loss function compares them with predicted probabilities. Shape (pixels,).
    labels = torch.tensor(table[target].to_numpy(), dtype=torch.float32)

    # cell_id rides along with every pixel, so predictions are matched back to
    # their place on the map by cell_id, never by row order
    cell_ids = torch.tensor(table["cell_id"].to_numpy())

    # Bundle the three: dataset[i] gives row i of each, so they stay together
    return TensorDataset(features, labels, cell_ids)


def batch_loader(dataset, batch_size, shuffle):
    """Hand out a dataset in mini-batches, one batch per step of a for loop.

    Parameters
    ----------
    dataset    : e.g. from pixel_dataset
    batch_size : pixels per batch; the last batch holds whatever is left over
    shuffle    : True for training (a new random order every epoch), False for
                 evaluation (the dataset's own order)

    Returns
    -------
    A DataLoader: `for features, labels, cell_ids in loader:` gives one batch per step.
    """
    # The order to visit the rows in: random, like sample(n) in R, or 0, 1, 2...
    # like seq_len(n). The random order is redrawn every epoch from torch's
    # random numbers, so torch.manual_seed() makes it repeatable.
    order = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    # Cut that order into chunks of batch_size rows, like
    # split(order, ceiling(seq_along(order) / batch_size)) in R.
    # drop_last=False keeps the last, shorter chunk, so no pixel is left out.
    chunks = BatchSampler(order, batch_size=batch_size, drop_last=False)

    # The DataLoader fetches each chunk's rows from the dataset in one go
    # (dataset[rows]). batch_size=None means "the chunks already are the
    # batches" (left at its default, it would wrap each chunk in an extra
    # batch dimension). A plain DataLoader(dataset, batch_size, shuffle=True)
    # fetches pixel by pixel and regroups them: about 10x slower (4.9 s
    # instead of 0.4 s per epoch on our data).
    return DataLoader(dataset, sampler=chunks, batch_size=None)
