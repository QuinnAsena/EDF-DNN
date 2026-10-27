"""PyTorch Datasets: different views of the same gridded data.

Why this module exists
----------------------
A PyTorch Dataset answers two questions: how many samples are there, and what
is sample number i? A DataLoader then draws mini-batches from it. Feeding
mini-batches, instead of the whole dataset at once (full-batch), is what lets
CNNs fit in memory and is standard on GPUs.

Planned contents (phases 3 and 4)
---------------------------------
- PixelDataset : one sample = one pixel's predictor values + its burn label
                 (the MLP's view; replaces the old CSV tables)
- PatchDataset : one sample = a square window of pixels around a location
                 (the CNN and U-Net view)
Each sample also carries its cell_id so predictions can be mapped back.
"""
