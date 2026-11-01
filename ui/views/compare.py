"""Compare tab: all five saved models side by side on the active csv."""

from __future__ import annotations

import streamlit as st

from shopper_intent.evaluation import leaderboard, pr_points, roc_points
from shopper_intent.models import MODELS

from .. import charts
from .. import components as ui
from ..state import ActiveData, evaluate_active, evaluate_all


def render(data: ActiveData) -> None:
    if not data.labelled:
        st.info("Model comparison needs the `Revenue` column in the uploaded csv.")
        return

    with st.spinner("Scoring all five models…"):
        results = evaluate_all(data)
    board = leaderboard(results)

    best = board.index[0]
    runner_up = board.index[1]
    gap = board.loc[best, "MCC"] - board.loc[runner_up, "MCC"]
    ui.winner_banner(
        best,
        f"Highest MCC ({board.loc[best, 'MCC']:.4f}), {gap:.4f} ahead of {runner_up}. "
        "MCC is the ranking metric because it stays honest on imbalanced data - "
        f"a model that never predicts 'Purchase' would still score {1 - data.y.mean():.1%} "
        "accuracy on this file, but MCC 0.",
    )

    ui.section("Leaderboard", f"All five saved models on <b>{data.name}</b> · sorted by MCC · "
                              "best value in each column highlighted")
    st.dataframe(
        board.style.format("{:.4f}").highlight_max(axis=0, props="font-weight:700; background-color:#e3eefc"),
        width="stretch",
    )

    ui.section("Metric by metric", "Same colour = same model on every chart")
    st.plotly_chart(charts.metric_bars(board), width="stretch")

    roc_curves, pr_curves, aucs = {}, {}, {}
    for key, spec in MODELS.items():
        _, _, scores = evaluate_active(key, data)
        roc_curves[spec.name] = roc_points(data.y, scores)
        pr_curves[spec.name] = pr_points(data.y, scores)
        aucs[spec.name] = results[spec.name].auc

    left, right = st.columns(2, gap="large")
    with left:
        ui.section("ROC curves", "Click a legend entry to hide a model")
        st.plotly_chart(charts.roc_curve_chart(roc_curves, aucs), width="stretch")
    with right:
        ui.section("Precision–recall curves", "Higher and further right is better")
        st.plotly_chart(charts.pr_curve_chart(pr_curves, float(data.y.mean())), width="stretch")

    st.download_button("Download leaderboard (.csv)", board.round(4).to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")
