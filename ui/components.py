"""Small HTML building blocks rendered with st.markdown."""

from __future__ import annotations

import math
from html import escape

import streamlit as st

from shopper_intent.evaluation import METRIC_LABELS

from . import theme as t

METRIC_HELP = {
    "accuracy": "Share of all sessions classified correctly",
    "auc": "Ranking quality across every threshold (0.5 = coin flip)",
    "precision": "Of sessions flagged as buyers, how many bought",
    "recall": "Of real buyers, how many were caught",
    "f1": "Harmonic mean of precision and recall",
    "mcc": "Correlation of predictions vs truth; robust to imbalance (-1..1)",
}


def hero(eyebrow: str, title: str, body: str, chips: dict[str, str] | None = None) -> None:
    chips_html = "".join(
        f'<span class="chip">{escape(k)} <b>{escape(v)}</b></span>' for k, v in (chips or {}).items()
    )
    st.markdown(
        f"""<div class="hero">
              <div class="eyebrow">{escape(eyebrow)}</div>
              <h1>{escape(title)}</h1>
              <p>{body}</p>
              <div class="chips">{chips_html}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="section-title">{escape(title)}</div>{sub}', unsafe_allow_html=True)


def callout(html: str) -> None:
    st.markdown(f'<div class="callout">{html}</div>', unsafe_allow_html=True)


def _fmt(value: float) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def metric_cards(metrics: dict[str, float], baseline: dict[str, float] | None = None,
                 baseline_label: str = "vs saved", compact: bool = False) -> None:
    """Six cards (one row, or 3x2 when compact). If a baseline is given, each card shows the delta."""
    cards = []
    for key, label in METRIC_LABELS.items():
        value = metrics.get(key)
        delta_html = ""
        if baseline and key in baseline and value is not None and not math.isnan(value):
            diff = value - baseline[key]
            css = "up" if diff > 0.0005 else "down" if diff < -0.0005 else ""
            delta_html = f'<div class="delta {css}">{diff:+.4f} {escape(baseline_label)}</div>'
        cards.append(
            f'<div class="metric-card" title="{escape(METRIC_HELP[key])}">'
            f'<div class="label">{label}</div><div class="value">{_fmt(value)}</div>{delta_html}</div>'
        )
    grid_class = "metric-grid compact" if compact else "metric-grid"
    st.markdown(f'<div class="{grid_class}">{"".join(cards)}</div>', unsafe_allow_html=True)


def stepper(titles: list[str]) -> None:
    """A row of completed stages joined by arrows."""
    items = "".join(
        f'<div class="step"><span class="step-dot">✓</span>{escape(title)}</div>' for title in titles
    )
    st.markdown(f'<div class="stepper">{items}</div>', unsafe_allow_html=True)


def model_swatch(name: str) -> str:
    color = t.MODEL_COLORS.get(name, t.ACCENT)
    return f'<span class="swatch" style="background:{color}"></span>{escape(name)}'


def winner_banner(name: str, reason: str) -> None:
    st.markdown(
        f'<div class="winner"><div class="section-sub" style="margin:0">Best on this data</div>'
        f'<div class="name">{model_swatch(name)}</div><div class="why">{reason}</div></div>',
        unsafe_allow_html=True,
    )
