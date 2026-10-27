"""The training loop, split into small named steps.

Planned contents (phase 3), each short enough to read on one screen:
- train_one_epoch : one pass over the training DataLoader, updating weights
- evaluate        : predictions on a DataLoader with gradients switched off
- EarlyStopping   : tracks the best validation loss and keeps those weights
- the fold loop   : for each fold, fit scaling, build loaders, train, evaluate

The per-fold scaling and class-imbalance handling from the old
run_grid_search_cv (geocomp-2025/src/train.py) carries over unchanged in logic.
All tensors go to a `device`, so the same code runs on CPU or GPU.
"""
