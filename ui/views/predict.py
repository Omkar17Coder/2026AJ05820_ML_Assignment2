"""Predict tab: describe one browsing session and see what every model thinks."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from shopper_intent import config as cfg
from shopper_intent.evaluation import positive_scores
from shopper_intent.models import MODELS

from .. import charts
from .. import components as ui
from ..state import get_models, get_train_frame

PRESETS = {
    "Typical session (training medians)": None,
    "Real buyer (training data)": 1,
    "Real non-buyer (training data)": 0,
}


def _starting_row(train: pd.DataFrame, preset_label: int | None) -> pd.Series:
    if preset_label is None:
        categorical = cfg.TEXT_CATEGORICAL_COLS + cfg.CODED_CATEGORICAL_COLS + cfg.BOOL_COLS
        return pd.Series({
            col: train[col].mode()[0] if col in categorical else train[col].median()
            for col in cfg.FEATURE_COLS
        })
    # a reproducible, typical example of that class: buyers who reached a valued page,
    # non-buyers who browsed a few products but never did
    reached_value = train["PageValues"] > 0
    typical = reached_value if preset_label == 1 else ~reached_value
    pool = train[(train[cfg.TARGET] == preset_label) & typical & (train["ProductRelated"] >= 5)]
    return pool.sample(1, random_state=cfg.RNG).iloc[0][cfg.FEATURE_COLS]


def render() -> None:
    train = get_train_frame()
    ui.section("Score a single session",
               "Fields start from a preset; change any of them and every saved model re-scores instantly.")

    preset = st.radio("Start from", list(PRESETS), horizontal=True, label_visibility="collapsed")
    start = _starting_row(train, PRESETS[preset])
    k = f"p{list(PRESETS).index(preset)}_"  # widget keys per preset, so switching resets the fields

    tab_pages, tab_ga, tab_ctx = st.columns(3, gap="large")
    row: dict = {}
    with tab_pages:
        st.markdown("**Pages visited & time spent**")
        for count_col, dur_col, label in [
            ("Administrative", "Administrative_Duration", "Account / admin"),
            ("Informational", "Informational_Duration", "Info / about"),
            ("ProductRelated", "ProductRelated_Duration", "Product"),
        ]:
            a, b = st.columns(2)
            row[count_col] = a.number_input(f"{label} pages", 0, 1000, int(start[count_col]), key=k + count_col)
            row[dur_col] = b.number_input(f"{label} seconds", 0.0, 100000.0, float(start[dur_col]),
                                          step=10.0, key=k + dur_col)
    with tab_ga:
        st.markdown("**Google Analytics signals**")
        row["PageValues"] = st.number_input(
            "Page value", 0.0, 1000.0, float(start["PageValues"]), step=1.0, key=k + "PageValues",
            help="Average value of pages visited before a transaction - the strongest single signal.",
        )
        row["BounceRates"] = st.slider("Bounce rate", 0.0, 0.2, float(start["BounceRates"]), 0.001,
                                       format="%.3f", key=k + "BounceRates")
        row["ExitRates"] = st.slider("Exit rate", 0.0, 0.2, float(start["ExitRates"]), 0.001,
                                     format="%.3f", key=k + "ExitRates")
        row["SpecialDay"] = st.select_slider(
            "Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], float(start["SpecialDay"]),
            key=k + "SpecialDay", help="1.0 = on the best shopping day before e.g. Valentine's Day",
        )
    with tab_ctx:
        st.markdown("**Visitor context**")
        months = [m for m in cfg.MONTH_ORDER if m in set(train["Month"])]
        row["Month"] = st.selectbox("Month", months, months.index(start["Month"]), key=k + "Month")
        row["VisitorType"] = st.selectbox("Visitor type", cfg.VISITOR_TYPES,
                                          cfg.VISITOR_TYPES.index(start["VisitorType"]), key=k + "VisitorType")
        row["Weekend"] = int(st.toggle("Weekend", bool(start["Weekend"]), key=k + "Weekend"))
        a, b = st.columns(2)
        for i, col in enumerate(cfg.CODED_CATEGORICAL_COLS):
            options = sorted(train[col].unique().tolist())
            row[col] = (a if i % 2 == 0 else b).selectbox(col, options, options.index(int(start[col])),
                                                          key=k + col)

    session = pd.DataFrame([row])[cfg.FEATURE_COLS]
    models = get_models()
    probs = {spec.name: float(positive_scores(models[key], session)[0]) for key, spec in MODELS.items()}
    votes = sum(p >= 0.5 for p in probs.values())

    left, right = st.columns([3, 2], gap="large", vertical_alignment="center")
    with left:
        st.plotly_chart(charts.probability_bars(probs), width="stretch")
    with right:
        verdict = "likely to buy" if votes >= 3 else "unlikely to buy"
        st.markdown(f"### {votes} of {len(probs)} models predict a purchase")
        st.markdown(f"Majority verdict: this visitor is **{verdict}**.")
        st.caption(f"Average P(purchase) across models: {sum(probs.values()) / len(probs):.1%}")
        if row["PageValues"] == 0:
            ui.callout("This session never reached a page with value. In training data only "
                       f"<b>{train.loc[train['PageValues'] == 0, cfg.TARGET].mean():.1%}</b> "
                       "of such sessions ended in a purchase - try raising <b>Page value</b>.")
