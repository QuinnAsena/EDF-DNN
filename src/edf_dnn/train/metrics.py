"""All evaluation metrics, computed in one place.

Planned contents (phase 3): one function taking true labels and predicted
probabilities and returning ROC AUC, average precision (the standard
PR-curve summary), Brier score, and recall/precision at a threshold.
Every model and every fold uses this same function.
"""
