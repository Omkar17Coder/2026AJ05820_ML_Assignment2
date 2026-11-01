"""Pipeline tab: watch the active csv move through the saved model, stage by stage."""

from __future__ import annotations

import time

import streamlit as st

from shopper_intent.features import LOG_SCALE_COLS, PLAIN_SCALE_COLS, SCALER_LABELS
from shopper_intent.models import MODELS
from shopper_intent.trace import trace_pipeline

from .. import charts
from .. import components as ui
from ..state import ActiveData, get_metadata, get_models

RESULT_KEY = "pipeline_trace"
# the real stages finish in milliseconds; this display-only pause keeps them readable
STEP_PAUSE_SECONDS = 0.35


def render(data: ActiveData, model_key: str) -> None:
    spec = MODELS[model_key]
    scaler = get_metadata().get("models", {}).get(model_key, {}).get("scaler")
    scaler_note = f" · preprocessing: <b>{SCALER_LABELS[scaler]}</b>" if scaler else ""
    ui.section("Processing pipeline",
               f"<b>{data.name}</b> through <b>{spec.name}</b>{scaler_note}. "
               "Change the file or model in the sidebar. Nothing is retrained here.")

    run_id = (data.digest, model_key)
    if st.button("▶ Run pipeline", type="primary", key="pipeline_run"):
        st.session_state[RESULT_KEY] = {"id": run_id, "steps": _run_live(data, model_key)}
        st.rerun()

    result = st.session_state.get(RESULT_KEY)
    if not result or result["id"] != run_id:
        st.caption("Press **Run pipeline** to process the file and see what each stage does.")
        return
    _render_result(result["steps"])


def _run_live(data: ActiveData, model_key: str) -> list:
    progress = st.progress(0.0, text="Starting…")
    total = 7 if data.labelled else 6
    steps = []
    for step in trace_pipeline(get_models()[model_key], data.content):
        steps.append(step)
        progress.progress(len(steps) / total, text=f"Step {len(steps)} of {total}: {step.title}")
        with st.status(step.title, state="complete"):
            st.write(step.detail)
        time.sleep(STEP_PAUSE_SECONDS)
    return steps


def _render_result(steps: list) -> None:
    ui.stepper([step.title for step in steps])
    total_ms = sum(step.seconds for step in steps) * 1000
    st.caption(f"All {len(steps)} stages finished in {total_ms:,.0f} ms of processing time.")

    for number, step in enumerate(steps, start=1):
        label = f"{number}. {step.title} · {step.rows:,} rows × {step.columns} columns · {step.seconds * 1000:,.1f} ms"
        with st.expander(label):
            st.markdown(step.detail)
            if step.sample is not None:
                st.dataframe(step.sample, width="stretch", hide_index=True)

    by_title = {step.title: step for step in steps}
    before = by_title["Engineer features"].extras["engineered"]
    after = by_title["Scale and encode"].extras["transformed"]

    ui.section("What scaling did to a feature", "The same rows before and after this model's fitted scaler")
    columns = LOG_SCALE_COLS + PLAIN_SCALE_COLS
    feature = st.selectbox("Feature", columns, index=columns.index("ProductRelated_Duration"), key="pipeline_feature")
    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(charts.value_histogram(before[feature], f"Before: {feature}"), width="stretch")
    with right:
        st.plotly_chart(charts.value_histogram(after[feature], f"After: {feature}"), width="stretch")
