"""The six assignment metrics plus the curves/tables the app draws."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config as cfg

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
}


@dataclass
class Metrics:
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def positive_scores(model, X) -> np.ndarray:
    """Score for the 'Purchase' class: probability when available, else margin."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:  # SVC built without probability=True
            pass
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> Metrics:
    """Precision / recall / F1 are reported for the positive (Purchase) class -
    with only ~15% buyers, that is the class we actually care about finding."""
    has_both_classes = len(np.unique(y_true)) == 2
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_score) if has_both_classes else float("nan"),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
    )


def predict_at(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def evaluate(model, X, y, threshold: float | None = None):
    """Returns (Metrics, y_pred, y_score). threshold=None uses the model's own predict()."""
    scores = positive_scores(model, X)
    y_pred = model.predict(X) if threshold is None else predict_at(scores, threshold)
    return compute_metrics(y, y_pred, scores), y_pred, scores


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=cfg.CLASS_NAMES, columns=cfg.CLASS_NAMES)


def report_frame(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=cfg.CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    frame = pd.DataFrame(report).T
    # sklearn stores accuracy as a bare float, which becomes a row with no support
    frame.loc["accuracy", ["precision", "recall"]] = float("nan")
    frame.loc["accuracy", "support"] = len(y_true)
    frame["support"] = frame["support"].astype(int)
    score_cols = ["precision", "recall", "f1-score"]
    frame[score_cols] = frame[score_cols].astype(float)
    return frame


def roc_points(y_true, y_score) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def pr_points(y_true, y_score) -> pd.DataFrame:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return pd.DataFrame({"recall": recall, "precision": precision})


def leaderboard(results: dict[str, Metrics]) -> pd.DataFrame:
    """Rows = model display name, columns = the six metrics, sorted by MCC."""
    frame = pd.DataFrame({name: m.as_dict() for name, m in results.items()}).T
    frame.index.name = "Model"
    return frame.rename(columns=METRIC_LABELS).sort_values("MCC", ascending=False)
