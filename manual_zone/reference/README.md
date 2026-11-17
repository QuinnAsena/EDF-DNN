# Reference copies from geocomp-2025

Read-only copies kept so EDF-DNN never needs the old repo. Not imported or run.

| File | Why it's kept |
|---|---|
| `train.py` | `run_grid_search_cv`: per-fold scaling and imbalance handling, early stopping, final retrain. Phase 3 ports this logic into `src/edf_dnn/train/`. |
| `data.py` | Old CSV loading, hold-out split, downsampling and `pos_weight` helpers. |
| `NOTES.md` | Old notes, including the k-fold design reasoning and spatial block CV notes. |
| `mlp_two_lyr_downsampled.qmd` | Plotting code for loss curves, confusion matrix and ROC curve, for the reports phase. |
