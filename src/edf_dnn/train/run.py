"""Entry point: train one experiment described by a config file.

Planned usage (phase 3):
    pixi run python -m edf_dnn.train.run configs/experiments/mlp_baseline.yaml

Reads the experiment config, trains with cross-validation, and writes
everything to runs/<experiment_name>/: the resolved config, the git commit,
scaling statistics, metrics and checkpoints. If that folder already exists,
the run stops unless the config sets `overwrite: true`.
"""
