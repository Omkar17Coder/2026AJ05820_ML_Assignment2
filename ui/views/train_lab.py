"""Training Lab: retrain one model with your own hyper-parameters, in-session.

Training always uses the bundled data/train_data.csv - never the uploaded test
file - and nothing is written to model/, so the saved models graded by the
Evaluate/Compare tabs are untouched.
"""

from __future__ import annotations

import io
import time

import joblib
import pandas as pd
import streamlit as st

from shopper_intent import config as cfg
from shopper_intent.data import features_target
from shopper_intent.evaluation import METRIC_LABELS, evaluate
from shopper_intent.features import DEFAULT_SCALER, SCALER_LABELS, SCALERS
from shopper_intent.models import MODELS, build_pipeline

from .. import components as ui
from ..state import ActiveData, evaluate_active, get_metadata, get_train_frame

HISTORY_KEY = "lab_history"
LOG_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.5, 1.0, 4.0, 8.0, 10.0, 100.0]
LOG_SMOOTHING = [10.0 ** p for p in range(-9, 2)]


def _param_widgets(key: str, saved: dict) -> dict:
    """Hyper-parameter controls for one model, pre-filled with the tuned values."""
    def pick(options, value, fallback):
        return value if value in options else fallback

    if key == "logistic_regression":
        return {
            "C": st.select_slider("C (inverse regularisation)", LOG_C, pick(LOG_C, saved.get("C"), 1.0)),
            "class_weight": st.radio("Class weight", [None, "balanced"], horizontal=True,
                                     index=[None, "balanced"].index(saved.get("class_weight")),
                                     format_func=lambda v: v or "none"),
        }
    if key == "decision_tree":
        unlimited = st.checkbox("Unlimited depth", saved.get("max_depth") is None)
        depth = None if unlimited else st.slider("Max depth", 1, 30, saved.get("max_depth") or 8)
        return {
            "max_depth": depth,
            "min_samples_leaf": st.slider("Min samples per leaf", 1, 100, saved.get("min_samples_leaf", 1)),
            "criterion": st.radio("Split criterion", ["gini", "entropy", "log_loss"], horizontal=True,
                                  index=["gini", "entropy", "log_loss"].index(saved.get("criterion", "gini"))),
            "class_weight": st.radio("Class weight", [None, "balanced"], horizontal=True,
                                     index=[None, "balanced"].index(saved.get("class_weight")),
                                     format_func=lambda v: v or "none"),
        }
    if key == "knn":
        return {
            "n_neighbors": st.slider("Neighbours (k)", 1, 101, saved.get("n_neighbors", 5), 2),
            "weights": st.radio("Vote weighting", ["uniform", "distance"], horizontal=True,
                                index=["uniform", "distance"].index(saved.get("weights", "uniform"))),
            "p": st.radio("Distance", [1, 2], horizontal=True, index=[1, 2].index(saved.get("p", 2)),
                          format_func={1: "Manhattan", 2: "Euclidean"}.get),
        }
    if key == "naive_bayes":
        return {"var_smoothing": st.select_slider(
            "Variance smoothing", LOG_SMOOTHING,
            pick(LOG_SMOOTHING, saved.get("var_smoothing"), 1e-9), format_func=lambda v: f"{v:.0e}")}
    if key == "svm":
        st.caption("Trained without probability calibration here to stay fast; AUC uses the SVM margin.")
        gamma_options = ["scale", 0.001, 0.01, 0.1]
        return {
            "C": st.select_slider("C (margin softness)", LOG_C, pick(LOG_C, saved.get("C"), 1.0)),
            "gamma": st.select_slider("Gamma (RBF width)", gamma_options,
                                      pick(gamma_options, saved.get("gamma"), "scale")),
            "class_weight": st.radio("Class weight", [None, "balanced"], horizontal=True,
                                     index=[None, "balanced"].index(saved.get("class_weight")),
                                     format_func=lambda v: v or "none"),
        }
    raise KeyError(key)


def render(data: ActiveData) -> None:
    ui.section("Training lab",
               "Retrain any model with your own hyper-parameters and see how it compares with the saved one.")
    ui.callout(f"Trains on <b>data/train_data.csv</b> ({len(get_train_frame()):,} sessions). "
               "Your uploaded file is only used for scoring. Saved models in <code>model/</code> "
               "are <b>never overwritten</b>.")

    if not data.labelled:
        st.info("The active file has no `Revenue` column, so lab runs can't be scored on it. "
                "Switch to the bundled test set in the sidebar.")
        return
    eval_data = data
    metadata = get_metadata().get("models", {})

    controls, results = st.columns([1, 2], gap="large")
    with controls:
        key = st.selectbox("Model", list(MODELS), format_func=lambda k: MODELS[k].name, key="lab_model")
        saved_model = metadata.get(key, {})
        with st.container(border=True):
            scaler_options = list(SCALERS)
            # keyed per model, so switching model resets it to that model's tuned scaler
            scaler = st.selectbox("Preprocessing", scaler_options, format_func=SCALER_LABELS.get,
                                  index=scaler_options.index(saved_model.get("scaler", DEFAULT_SCALER)),
                                  key=f"lab_scaler_{key}")
            params = _param_widgets(key, saved_model.get("best_params", {}))
        train_clicked = st.button("Train model", type="primary", width="stretch")

    if train_clicked:
        train_df = get_train_frame()
        X_train, y_train = features_target(train_df)
        with st.spinner(f"Training {MODELS[key].name} on {len(X_train):,} sessions…"):
            start = time.perf_counter()
            pipe = build_pipeline(key, scaler=scaler, **params).fit(X_train, y_train)
            seconds = time.perf_counter() - start
        lab_metrics, _, _ = evaluate(pipe, eval_data.X, eval_data.y)
        saved_metrics, _, _ = evaluate_active(key, eval_data)

        buffer = io.BytesIO()
        joblib.dump(pipe, buffer, compress=3)
        st.session_state.setdefault(HISTORY_KEY, []).insert(0, {
            "model": MODELS[key].name, "key": key, "params": {"scaler": scaler, **params}, "seconds": seconds,
            "metrics": lab_metrics.as_dict(), "saved": saved_metrics.as_dict(),
            "evaluated_on": eval_data.name, "artifact": buffer.getvalue(),
        })

    history = st.session_state.get(HISTORY_KEY, [])
    with results:
        if not history:
            st.markdown("#### No runs yet")
            st.caption("Pick a model, adjust its knobs and press **Train model**. "
                       "Controls start at the values found by grid search.")
            return

        latest = history[0]
        st.markdown(f"#### Latest run · {latest['model']}")
        st.caption(f"{_fmt_params(latest['params'])} · trained in {latest['seconds']:.1f}s · "
                   f"scored on {latest['evaluated_on']}")
        ui.metric_cards(latest["metrics"], baseline=latest["saved"], baseline_label="vs saved", compact=True)
        st.download_button("Download this model (.joblib)", latest["artifact"],
                           file_name=f"lab_{latest['key']}.joblib", width="stretch")

    ui.section("Run history", "Every run in this browser session, newest first")
    rows = [{"Model": r["model"], "Params": _fmt_params(r["params"]), "Seconds": r["seconds"],
             **{METRIC_LABELS[m]: v for m, v in r["metrics"].items()},
             "Δ MCC vs saved": r["metrics"]["mcc"] - r["saved"]["mcc"]} for r in history]
    frame = pd.DataFrame(rows)
    st.dataframe(frame.style.format({c: "{:.4f}" for c in list(METRIC_LABELS.values())}
                                    | {"Δ MCC vs saved": "{:+.4f}", "Seconds": "{:.1f}"}),
                 width="stretch", hide_index=True)
    if st.button("Clear history"):
        st.session_state[HISTORY_KEY] = []
        st.rerun()


def _fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v if v is not None else 'none'}" for k, v in params.items())
