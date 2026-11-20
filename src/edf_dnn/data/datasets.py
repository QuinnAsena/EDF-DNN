"""PyTorch datasets and loaders: how pixels reach the network in mini-batches.

Contents
--------
- pixel_dataset : pixel table -> TensorDataset of (features, labels, cell_ids),
                  one sample per pixel (the MLP's view; replaces the old to_tensors).
- batch_loader  : hands a dataset out in mini-batches, shuffled for training
                  or in order for evaluation.
- BatchLoader   : what batch_loader returns: cuts batches straight from the
                  dataset's tensors, on whatever device they live on.

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
                 its __getitem__ must accept a tensor of row numbers and return
                 the whole batch (batch_loader asks for dataset[rows]).
"""

import math

import torch
from torch.utils.data import TensorDataset


def pixel_dataset(table, variables, scaler, target="burned01", device="cpu"):
    """Turn a pixel table into a PyTorch dataset with one sample per pixel.

    Parameters
    ----------
    table     : pixel table (rows of to_pixel_table), already split: e.g. the
                training pixels of one fold, or its validation pixels
    variables : names of the predictor columns, as the scaler was fitted on
    scaler    : MinMaxScaler fitted on the TRAINING pixels (see normalise.py)
    target    : name of the label column
    device    : where the tensors live: "cpu", or the training device (e.g.
                choose_device()). On a GPU, putting the whole dataset there once
                saves copying every batch from the CPU, which takes most of an
                epoch for a model this small (see the Benchmarks section of the
                README). The pixel tables are small enough (a 563,475-pixel
                table with 25 predictors is about 60 MB) to fit easily.

    Returns
    -------
    A TensorDataset: dataset[i] is (features, label, cell_id) for row i of the
    table, and len(dataset) is the number of pixels. Every pixel is kept,
    burned and unburned; nothing is filtered.
    """
    # Predictors: scaled to 0-1 (sklearn checks the names match the fit), as
    # float32, the precision networks compute in. Shape (pixels, variables).
    features = torch.tensor(scaler.transform(table[variables]), dtype=torch.float32, device=device)

    # Labels: the whole 0/1 column, like table$burned01 in R. float32 because
    # the loss function compares them with predicted probabilities. Shape (pixels,).
    labels = torch.tensor(table[target].to_numpy(), dtype=torch.float32, device=device)

    # cell_id rides along with every pixel, so predictions are matched back to
    # their place on the map by cell_id, never by row order
    cell_ids = torch.tensor(table["cell_id"].to_numpy(), device=device)

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
    A BatchLoader: `for features, labels, cell_ids in loader:` gives one batch
    per step, and len(loader) is the number of batches per epoch.
    """
    return BatchLoader(dataset, batch_size, shuffle)


class BatchLoader:
    """Deals a dataset out in mini-batches, cut straight from its tensors.

    Why not torch's DataLoader: with a sampler, DataLoader hands the dataset a
    Python list of row numbers for every batch, and the rows are gathered on the
    CPU. For a small network that bookkeeping, not the network, takes most of
    the epoch (see the README's Benchmarks). Here the random order is one tensor,
    drawn on the same device as the data, and each batch is a slice of it used
    as an index: one fast gather per tensor, on the GPU if the data live there.

    Reproducibility: the order comes from torch's random numbers on the data's
    device (torch.randperm, like sample(n) in R), redrawn every epoch, so
    torch.manual_seed() makes it repeatable.
    """

    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        # Number of batches per epoch, counting the last, shorter one
        return math.ceil(len(self.dataset) / self.batch_size)

    def _device(self):
        # A TensorDataset's tensors carry their device; other datasets (the
        # planned PatchDataset) can set a .device attribute, else the CPU
        if isinstance(self.dataset, TensorDataset):
            return self.dataset.tensors[0].device
        return getattr(self.dataset, "device", torch.device("cpu"))

    def _take(self, rows):
        # A TensorDataset: index each of its tensors with the batch's rows. Any
        # other dataset is asked for the whole batch at once: dataset[rows]
        if isinstance(self.dataset, TensorDataset):
            return tuple(tensor[rows] for tensor in self.dataset.tensors)
        return self.dataset[rows]

    def __iter__(self):
        pixel_count = len(self.dataset)
        device = self._device()
        # The order to visit the rows in: random, like sample(n) in R, or
        # 0, 1, 2... like seq_len(n), as a tensor on the data's device
        if self.shuffle:
            order = torch.randperm(pixel_count, device=device)
        else:
            order = torch.arange(pixel_count, device=device)
        # Cut that order into chunks of batch_size rows, keeping the last,
        # shorter chunk, so no pixel is left out
        for start in range(0, pixel_count, self.batch_size):
            yield self._take(order[start:start + self.batch_size])
