"""Evaluation utilities for classification models."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true,
    y_pred,
    y_prob=None,
) -> dict[str, float]:
    """Calculate classification metrics with flop as the positive class."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "flop_precision": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "flop_recall": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "flop_f1": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def metrics_to_dataframe(
    model_name: str,
    metrics: dict[str, float],
) -> pd.DataFrame:
    """Convert one model's metric dictionary into a table row."""

    return pd.DataFrame(
        [{"model": model_name, **metrics}]
    )


def get_confusion_matrix(y_true, y_pred):
    """Return the confusion matrix using hit first and flop second."""

    return confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )