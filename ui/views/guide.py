"""Start here tab: what this project does and where the reviewer finds each requirement."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from shopper_intent.evaluation import Metrics, leaderboard

from .. import components as ui
from ..state import get_saved_metrics

REQUIREMENTS = [
    ("Dataset upload (CSV)", "Sidebar → 1 · Data → Upload my own CSV"),
    ("Model selection dropdown", "Sidebar → 2 · Model"),
    ("Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)", "Evaluate tab, metric cards"),
    ("Confusion matrix and classification report", "Evaluate tab"),
    ("Results of all models on the test data", "Compare models tab"),
    ("Saved models are loaded, not retrained on the upload", "Pipeline tab shows each stage; files are in model/"),
]

EXTRAS = [
    ("Pipeline", "Runs the active file through the saved model stage by stage, with counts, timings and scaling charts."),
    ("Evaluate → threshold slider", "Recomputes the metrics live, to show the precision and recall trade-off."),
    ("Predict a session", "Fill in one browsing session and see what all five models predict."),
    ("Training lab", "Retrain a model with your own hyper-parameters and scaler. Saved models are never overwritten."),
    ("About", "Problem statement, data dictionary, EDA charts, architecture diagram and tuned settings."),
]


def render() -> None:
    ui.section("What this project does")
    st.markdown(
        "This app predicts whether an online shopping session ends in a purchase, using the UCI "
        "*Online Shoppers Purchasing Intention* dataset. Five classifiers were trained and tuned offline: "
        "Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM. Each one is saved with its "
        "preprocessing, and this app loads those saved models to score test data."
    )

    ui.section("Try it in four steps")
    st.markdown(
        "1. **Choose the data.** The bundled `test_data.csv` is already selected. To upload your own, "
        "download the sample from the sidebar and choose *Upload my own CSV*.\n"
        "2. **Pick a model** from the sidebar dropdown.\n"
        "3. **Open Evaluate** for its metrics, confusion matrix and classification report.\n"
        "4. **Open Compare models** to see all five side by side, or **Pipeline** to watch the processing."
    )

    ui.section("Where each requirement is")
    st.dataframe(pd.DataFrame(REQUIREMENTS, columns=["Assignment requirement", "Where to find it"]),
                 width="stretch", hide_index=True)

    ui.section("Added features")
    st.dataframe(pd.DataFrame(EXTRAS, columns=["Feature", "What it does"]), width="stretch", hide_index=True)

    saved = get_saved_metrics()
    if saved:
        board = leaderboard({name: Metrics(**values) for name, values in saved.items()})
        ui.section("Results on the held-out test set", "Saved at training time · sorted by MCC")
        st.dataframe(board.style.format("{:.4f}"), width="stretch")
