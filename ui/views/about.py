"""About tab: problem statement, data dictionary, EDA highlights and how the pipeline fits together."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from shopper_intent import config as cfg
from shopper_intent.features import SCALER_LABELS
from shopper_intent.models import MODELS

from .. import charts
from .. import components as ui
from ..state import get_metadata, get_train_frame

FEATURE_DICTIONARY = [
    ("Administrative", "int", "Account/admin pages visited in the session"),
    ("Administrative_Duration", "float", "Seconds spent on admin pages"),
    ("Informational", "int", "Info pages visited (about us, contact, …)"),
    ("Informational_Duration", "float", "Seconds spent on info pages"),
    ("ProductRelated", "int", "Product pages visited"),
    ("ProductRelated_Duration", "float", "Seconds spent on product pages"),
    ("BounceRates", "float", "Avg. % of visitors who left after entering on these pages"),
    ("ExitRates", "float", "Avg. % of page views that were the last in a session"),
    ("PageValues", "float", "Avg. value of pages visited before a transaction"),
    ("SpecialDay", "float", "Closeness to a holiday such as Valentine's or Mother's Day (0-1)"),
    ("Month", "category", "Month of the visit (10 months present - no Jan / Apr)"),
    ("OperatingSystems", "category", "Operating-system code"),
    ("Browser", "category", "Browser code"),
    ("Region", "category", "Geographic region code"),
    ("TrafficType", "category", "Traffic source code (ad, direct, referral, …)"),
    ("VisitorType", "category", "Returning, new or other visitor"),
    ("Weekend", "bool", "Session happened on a weekend"),
    ("Revenue", "bool · target", "Session ended in a purchase"),
]

PIPELINE_DOT = """
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.32; compound=true;
  node [shape=box, style="rounded,filled", fontname="Inter", fontsize=12,
        color="#d6d5d0", fillcolor="#ffffff", fontcolor="#0b0b0b", margin="0.22,0.10", width=2.6];
  edge [color="#8a8984", arrowsize=0.7, fontname="Inter", fontsize=10, fontcolor="#52514e"];

  subgraph cluster_offline {
    label="OFFLINE  ·  python -m shopper_intent.train"; labeljust="l"; fontname="Inter"; fontsize=11;
    fontcolor="#2a78d6"; style="rounded,dashed"; color="#86b6ef"; margin=14;

    raw   [label="UCI csv\\n12,330 sessions · 17 features"];
    clean [label="clean + dedupe\\nJune→Jun · TRUE/FALSE→1/0 · −125 dupes"];
    split [label="stratified 80 / 20 split\\n15.6% buyers in both halves"];
    gs    [label="GridSearchCV per model\\n5-fold · scored on MCC", fillcolor="#e8f1fc", color="#86b6ef"];
    art   [label="model/*.joblib\\n5 fitted pipelines + metrics.json", fillcolor="#e8f1fc", color="#86b6ef"];

    subgraph cluster_pipe {
      label="one sklearn Pipeline (saved whole)"; labeljust="l"; fontname="Inter"; fontsize=10;
      fontcolor="#52514e"; style="rounded"; color="#d6d5d0"; margin=10;
      fe  [label="SessionFeatureBuilder\\n+TotalPages · TotalDuration\\n+AvgSecondsPerPage · HasPageValue"];
      ct  [label="ColumnTransformer\\nbest scaler per model · one-hot"];
      clf [label="classifier\\nLR · Tree · KNN · NB · SVM"];
      fe -> ct -> clf;
    }
    raw -> clean -> split;
    split -> fe [label=" train 80%"];
    clf -> gs -> art;
  }

  subgraph cluster_online {
    label="ONLINE  ·  streamlit run app.py"; labeljust="l"; fontname="Inter"; fontsize=11;
    fontcolor="#1a7f4e"; style="rounded,dashed"; color="#9fd8bd"; margin=14;
    test   [label="test_data.csv  /  your upload"];
    check  [label="schema check + clean\\nclear error if a column is missing"];
    load   [label="load saved pipeline\\n(cached · never refit)"];
    result [label="metrics · confusion matrix\\nROC / PR · compare · predict", fillcolor="#2a78d6", fontcolor="#ffffff", color="#2a78d6"];
    test -> check -> load -> result;
  }

  split -> test [style=dashed, label=" held-out 20%"];
  art -> load [style=dashed];
}
"""


def render() -> None:
    train = get_train_frame()
    meta = get_metadata()

    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section("Problem statement")
        st.markdown(
            "An online store sees thousands of browsing sessions a day, but only about **one in "
            "six** ends in a purchase. Given what a visitor has done during a session - which pages "
            "they opened, how long they stayed, how 'valuable' those pages are according to Google "
            "Analytics, and when they visited - **predict whether the session will end in a purchase** "
            "(`Revenue = True`). A good model lets the business target live discounts or chat support "
            "at sessions that are close to converting."
        )
        ui.section("Dataset")
        st.markdown(
            f"[UCI Online Shoppers Purchasing Intention]({cfg.DATASET_URL}) (Sakar et al., 2018) - "
            "12,330 sessions from an e-commerce site over one year, each from a different user "
            "to avoid bias toward a single shopper. **17 features** (10 numeric, 7 categorical) plus "
            "the binary target. No missing values; 125 exact duplicate rows were removed."
        )
    with right:
        ui.section("Class balance", "Training split")
        st.plotly_chart(charts.class_balance(train[cfg.TARGET].value_counts().sort_index()), width="stretch")
        ui.callout("Imbalanced 84 / 16 - which is why <b>MCC</b> and <b>F1</b> matter more "
                   "than accuracy here, and why the split is stratified.")

    ui.section("How it fits together", "The same pipeline object is trained offline and loaded by this app")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")

    ui.section("What drives purchases", "Purchase rate per category in the training split · dotted line = overall rate")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**By month**")
        st.plotly_chart(charts.feature_rate_by(train, "Month", cfg.MONTH_ORDER), width="stretch")
    with b:
        st.markdown("**By visitor type**")
        st.plotly_chart(charts.feature_rate_by(train, "VisitorType", cfg.VISITOR_TYPES), width="stretch")

    ui.section("Data dictionary")
    st.dataframe(pd.DataFrame(FEATURE_DICTIONARY, columns=["Column", "Type", "Meaning"]),
                 width="stretch", hide_index=True, height=35 * (len(FEATURE_DICTIONARY) + 1) + 3)

    if meta:
        ui.section("Saved models", f"Trained {meta['trained_at']} · scikit-learn {meta['sklearn']} · "
                                   f"{meta['cv_folds']}-fold CV · seed {meta['random_state']}")
        rows = [{
            "Model": spec.name,
            "Preprocessing": SCALER_LABELS.get(meta["models"][key].get("scaler"), "default"),
            "Best hyper-parameters": ", ".join(f"{k}={v}" for k, v in
                                               meta["models"][key]["best_params"].items()) or "defaults",
            "CV MCC": meta["models"][key]["cv_mcc"],
            "Train time (s)": meta["models"][key]["fit_seconds"],
        } for key, spec in MODELS.items() if key in meta["models"]]
        st.dataframe(pd.DataFrame(rows).style.format({"CV MCC": "{:.4f}", "Train time (s)": "{:.1f}"}),
                     width="stretch", hide_index=True)
