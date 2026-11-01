"""Evaluate tab: one saved model on the active csv - metrics, confusion matrix,
classification report, ROC/PR curves and a live decision-threshold slider."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from shopper_intent import config as cfg
from shopper_intent.evaluation import confusion_frame, pr_points, report_frame, roc_points
from shopper_intent.models import MODELS

from .. import charts
from .. import components as ui
from ..state import ActiveData, evaluate_active

DEFAULT_THRESHOLD = 0.50


def render(data: ActiveData, model_key: str) -> None:
    spec = MODELS[model_key]

    if not data.labelled:
        _render_unlabelled(data, model_key)
        return

    head_left, head_right = st.columns([3, 2], vertical_alignment="bottom")
    with head_left:
        ui.section(f"{spec.name} on {data.name}",
                   f"{len(data.X):,} sessions · loaded from <code>model/{spec.key}.joblib</code> "
                   f"· no retraining")
    with head_right:
        threshold = st.slider(
            "Decision threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01, key="threshold",
            help="A session is predicted as 'Purchase' when P(purchase) ≥ threshold. "
                 "Lower it to catch more buyers (recall ↑), raise it to be more selective (precision ↑).",
        )

    at_default = abs(threshold - DEFAULT_THRESHOLD) < 1e-9
    default_metrics, _, _ = evaluate_active(model_key, data)
    metrics, y_pred, scores = evaluate_active(model_key, data, None if at_default else threshold)

    ui.metric_cards(metrics.as_dict(),
                    baseline=None if at_default else default_metrics.as_dict(),
                    baseline_label="vs 0.50")
    if not at_default:
        ui.callout(f"Metrics recomputed at threshold <b>{threshold:.2f}</b>. "
                   "AUC does not change - it already averages over every threshold.")

    cm = confusion_frame(data.y, y_pred)
    left, right = st.columns(2, gap="large")
    with left:
        ui.section("Confusion matrix", "Colour = share of each true class · hover for counts")
        st.plotly_chart(charts.confusion_heatmap(cm), width="stretch")
    with right:
        ui.section("Classification report", "Per-class precision, recall and F1")
        report = report_frame(data.y, y_pred)
        st.dataframe(
            report.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                 "f1-score": "{:.4f}", "support": "{:,}"}, na_rep=""),
            width="stretch",
        )
        tn, fp, fn, tp = cm.to_numpy().ravel()
        ui.callout(
            f"Caught <b>{tp:,}</b> of {tp + fn:,} buyers, missed <b>{fn:,}</b>. "
            f"<b>{fp:,}</b> non-buyers were wrongly flagged as buyers."
        )

    # operating point of the current threshold on both curves
    fpr = fp / max(fp + tn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    left, right = st.columns(2, gap="large")
    with left:
        ui.section("ROC curve", "How well the model ranks buyers above non-buyers")
        st.plotly_chart(
            charts.roc_curve_chart({spec.name: roc_points(data.y, scores)}, {spec.name: metrics.auc},
                                   marker=(fpr, recall)),
            width="stretch",
        )
    with right:
        ui.section("Precision–recall curve", "More informative than ROC when buyers are rare")
        st.plotly_chart(
            charts.pr_curve_chart({spec.name: pr_points(data.y, scores)}, float(data.y.mean()),
                                  marker=(recall, precision)),
            width="stretch",
        )

    ui.section("Where the threshold cuts",
               "Each class's predicted purchase probability, normalised so both are visible despite the 84/16 imbalance")
    st.plotly_chart(charts.score_distribution(scores, data.y.to_numpy(), threshold), width="stretch")

    _download_predictions(data, model_key, y_pred, scores)


def _render_unlabelled(data: ActiveData, model_key: str) -> None:
    spec = MODELS[model_key]
    st.info(f"**{data.name}** has no `{cfg.TARGET}` column, so metrics can't be computed. "
            f"Showing {spec.name} predictions instead - add the target column to see the full evaluation.")
    _, y_pred, scores = evaluate_active(model_key, data)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions", f"{len(y_pred):,}")
    c2.metric("Predicted buyers", f"{int(y_pred.sum()):,}")
    c3.metric("Predicted purchase rate", f"{y_pred.mean():.1%}")
    _download_predictions(data, model_key, y_pred, scores)


def _download_predictions(data: ActiveData, model_key: str, y_pred, scores) -> None:
    ui.section("Predictions", "Row-level output for the active file")
    out = data.raw.copy()
    out.insert(0, "predicted_label", pd.Series(y_pred).map(dict(enumerate(cfg.CLASS_NAMES))).to_numpy())
    out.insert(1, "purchase_probability", scores.round(4))
    st.dataframe(out.head(200), width="stretch", height=280)
    st.download_button(
        "Download predictions (.csv)", out.to_csv(index=False).encode(),
        file_name=f"predictions_{model_key}.csv", mime="text/csv",
    )
