"""Runs a csv through a saved pipeline one stage at a time and reports each stage.

The stages call the fitted objects inside the pipeline with transform/predict only,
so what gets reported is exactly what `model.predict` does - nothing is refitted.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from . import config as cfg
from .data import check_schema, clean
from .evaluation import compute_metrics, positive_scores


@dataclass
class TraceStep:
    title: str
    detail: str
    rows: int
    columns: int
    seconds: float
    sample: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)


def _sample(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return frame.head(n).reset_index(drop=True)


def trace_pipeline(model, content: bytes) -> Iterator[TraceStep]:
    """Yields one TraceStep as each stage finishes, so a UI can show live progress."""
    start = time.perf_counter()
    raw = pd.read_csv(io.BytesIO(content))
    yield TraceStep("Read CSV", "Parsed the file into a table.", len(raw), raw.shape[1],
                    time.perf_counter() - start, _sample(raw))

    start = time.perf_counter()
    report = check_schema(raw)
    target_note = f"`{cfg.TARGET}` found, so metrics can be computed." if report.has_target \
        else f"No `{cfg.TARGET}` column, so only predictions are possible."
    yield TraceStep("Check schema",
                    f"All {len(cfg.FEATURE_COLS)} required features are present. {target_note}",
                    len(raw), raw.shape[1], time.perf_counter() - start)

    start = time.perf_counter()
    cleaned = clean(raw)
    months_fixed = int((raw["Month"].astype(str) != cleaned["Month"]).sum())
    y = cleaned[cfg.TARGET].astype(int).to_numpy() if report.has_target else None
    X = cleaned[cfg.FEATURE_COLS]
    yield TraceStep("Clean", f"Month spellings normalised in {months_fixed:,} rows; true/false values mapped to 1/0. "
                             "Target set aside.",
                    len(X), X.shape[1], time.perf_counter() - start, _sample(X))

    prep, clf = model.named_steps["prep"], model.named_steps["clf"]

    start = time.perf_counter()
    engineered = prep.named_steps["engineer"].transform(X)
    added = [c for c in engineered.columns if c not in X.columns]
    yield TraceStep("Engineer features", "Added " + ", ".join(added) + ".",
                    len(engineered), engineered.shape[1], time.perf_counter() - start, _sample(engineered),
                    {"engineered": engineered})

    start = time.perf_counter()
    columns = prep.named_steps["columns"]
    transformed = columns.transform(engineered)
    names = list(columns.get_feature_names_out())
    transformed_frame = pd.DataFrame(transformed, columns=names, index=engineered.index)
    yield TraceStep("Scale and encode",
                    f"Numeric columns rescaled and categories one-hot encoded: "
                    f"{engineered.shape[1]} columns became {len(names)} model inputs.",
                    transformed.shape[0], transformed.shape[1], time.perf_counter() - start,
                    _sample(transformed_frame).round(3), {"transformed": transformed_frame})

    start = time.perf_counter()
    y_pred = clf.predict(transformed)
    scores = positive_scores(clf, transformed)
    predictions = pd.DataFrame({"predicted": y_pred, "purchase_probability": np.round(scores, 4)})
    yield TraceStep("Predict", f"Flagged {int(y_pred.sum()):,} of {len(y_pred):,} sessions as purchases.",
                    len(y_pred), predictions.shape[1], time.perf_counter() - start, _sample(predictions),
                    {"y_pred": y_pred})

    if y is None:
        return
    start = time.perf_counter()
    metrics = compute_metrics(y, y_pred, scores)
    yield TraceStep("Compute metrics", f"Accuracy {metrics.accuracy:.4f}, AUC {metrics.auc:.4f}, "
                                       f"F1 {metrics.f1:.4f}, MCC {metrics.mcc:.4f}.",
                    len(y), 6, time.perf_counter() - start, None, {"metrics": metrics})
