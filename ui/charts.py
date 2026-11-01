"""Plotly figures used across the app. Every chart goes through `_base_layout`
so fonts, grid and hover styling stay consistent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shopper_intent import config as cfg

from . import theme as t

_METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def _base_layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=t.FONT_FAMILY, color=t.INK_SECONDARY, size=12),
        hoverlabel=dict(bgcolor=t.SURFACE_RAISED, bordercolor=t.GRID, font=dict(color=t.INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=t.GRID, linecolor=t.GRID, zeroline=False, tickfont=dict(color=t.INK_SECONDARY))
    fig.update_yaxes(gridcolor=t.GRID, linecolor=t.GRID, zeroline=False, tickfont=dict(color=t.INK_SECONDARY))
    return fig


def confusion_heatmap(cm: pd.DataFrame) -> go.Figure:
    """Cell colour = share of the true class (row-normalised), label = raw count."""
    values = cm.to_numpy()
    total = values.sum()
    row_share = values / values.sum(axis=1, keepdims=True).clip(min=1)
    x_labels = [f"Pred: {c}" for c in cm.columns]
    y_labels = [f"True: {c}" for c in cm.index]
    colorscale = [[i / (len(t.SEQUENTIAL) - 1), c] for i, c in enumerate(t.SEQUENTIAL)]

    fig = go.Figure(go.Heatmap(
        z=row_share, x=x_labels, y=y_labels, colorscale=colorscale, zmin=0, zmax=1,
        showscale=False, xgap=3, ygap=3, customdata=values,
        hovertemplate="%{y} · %{x}<br>%{customdata:,} sessions (%{z:.1%} of row)<extra></extra>",
    ))
    for i, y_label in enumerate(y_labels):
        for j, x_label in enumerate(x_labels):
            share = row_share[i, j]
            fig.add_annotation(
                x=x_label, y=y_label, showarrow=False,
                text=f"<b>{values[i, j]:,}</b><br>{share:.1%} of row",
                font=dict(size=14, color="white" if share > 0.55 else t.INK),  # keep text readable on dark cells
            )
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, height=330, title=dict(text=f"{total:,} sessions", font=dict(size=12)))


def roc_curve_chart(curves: dict[str, pd.DataFrame], aucs: dict[str, float],
                    marker: tuple[float, float] | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(color=t.INK_MUTED, dash="dot", width=1), hoverinfo="skip"))
    for name, pts in curves.items():
        fig.add_trace(go.Scatter(
            x=pts["fpr"], y=pts["tpr"], mode="lines", name=f"{name} · AUC {aucs[name]:.3f}",
            line=dict(color=t.MODEL_COLORS.get(name, t.ACCENT), width=2),
            hovertemplate=f"{name}<br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>",
        ))
    if marker:
        fig.add_trace(_operating_point(*marker))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate (recall)", range=[0, 1.02])
    return _base_layout(fig, height=380, hovermode="closest")


def pr_curve_chart(curves: dict[str, pd.DataFrame], base_rate: float,
                   marker: tuple[float, float] | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=base_rate, line=dict(color=t.INK_MUTED, dash="dot", width=1),
                  annotation_text=f"base rate {base_rate:.1%}", annotation_font_color=t.INK_MUTED)
    for name, pts in curves.items():
        fig.add_trace(go.Scatter(
            x=pts["recall"], y=pts["precision"], mode="lines", name=name,
            line=dict(color=t.MODEL_COLORS.get(name, t.ACCENT), width=2),
            hovertemplate=f"{name}<br>Recall %{{x:.3f}} · Precision %{{y:.3f}}<extra></extra>",
        ))
    if marker:
        fig.add_trace(_operating_point(*marker))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _base_layout(fig, height=380, hovermode="closest")


def _operating_point(x: float, y: float) -> go.Scatter:
    return go.Scatter(
        x=[x], y=[y], mode="markers", name="Current threshold",
        marker=dict(size=11, color=t.INK, line=dict(color=t.SURFACE_RAISED, width=2)),
        hovertemplate="Current threshold<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
    )


def score_distribution(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> go.Figure:
    fig = go.Figure()
    bins = dict(start=0, end=1.0001, size=0.025)
    for label, name in enumerate(cfg.CLASS_NAMES):
        fig.add_trace(go.Histogram(
            x=scores[y_true == label], name=name, xbins=bins, opacity=0.75, histnorm="percent",
            marker=dict(color=t.CLASS_COLORS[name], line=dict(color=t.SURFACE, width=1)),
            hovertemplate=f"{name}<br>score %{{x}}<br>%{{y:.1f}}% of {name.lower()} sessions<extra></extra>",
        ))
    fig.add_vline(x=threshold, line=dict(color=t.INK, width=2, dash="dash"),
                  annotation_text=f"threshold {threshold:.2f}", annotation_position="top")
    fig.update_xaxes(title="Predicted purchase probability", range=[0, 1])
    fig.update_yaxes(title="Share of class (%)")
    return _base_layout(fig, height=330, barmode="overlay", bargap=0.04)


def metric_bars(board: pd.DataFrame) -> go.Figure:
    """Grouped bars: x = metric, one bar per model, colour = model."""
    fig = go.Figure()
    for name in [n for n in t.MODEL_COLORS if n in board.index]:
        row = board.loc[name]
        fig.add_trace(go.Bar(
            x=_METRIC_ORDER, y=[row[m] for m in _METRIC_ORDER], name=name,
            marker=dict(color=t.MODEL_COLORS[name], line=dict(color=t.SURFACE, width=2),
                        cornerradius=4),
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_yaxes(range=[0, 1.05], title=None)
    return _base_layout(fig, height=400, barmode="group", bargap=0.22, bargroupgap=0.06)


def probability_bars(probs: dict[str, float], threshold: float = 0.5) -> go.Figure:
    names = list(probs)
    values = [probs[n] for n in names]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=[t.MODEL_COLORS.get(n, t.ACCENT) for n in names], cornerradius=4),
        text=[f"{v:.1%}" for v in values], textposition="outside",
        textfont=dict(color=t.INK),
        hovertemplate="%{y}<br>P(purchase) = %{x:.1%}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line=dict(color=t.INK_MUTED, dash="dot", width=1),
                  annotation_text="0.5", annotation_font_color=t.INK_MUTED)
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", title="Probability of purchase")
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, height=300, showlegend=False)


def value_histogram(values: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=40, marker=dict(color=t.ACCENT, line=dict(color=t.SURFACE, width=1)),
        hovertemplate="%{x}<br>%{y} sessions<extra></extra>",
    ))
    fig.update_yaxes(title="Sessions")
    return _base_layout(fig, height=300, showlegend=False, bargap=0.04,
                        title=dict(text=f"{title} (skew {values.skew():.2f})", font=dict(size=13)))


def class_balance(counts: pd.Series) -> go.Figure:
    names = [cfg.CLASS_NAMES[i] for i in counts.index]
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.values, y=names, orientation="h",
        marker=dict(color=[t.CLASS_COLORS[n] for n in names], cornerradius=4),
        text=[f"{v:,} ({v / total:.1%})" for v in counts.values], textposition="auto",
        hovertemplate="%{y}: %{x:,} sessions<extra></extra>",
    ))
    fig.update_xaxes(title="Sessions")
    return _base_layout(fig, height=200, showlegend=False)


def feature_rate_by(df: pd.DataFrame, column: str, order: list | None = None) -> go.Figure:
    """Purchase rate per category, with session counts in the hover."""
    grouped = df.groupby(column)[cfg.TARGET].agg(["mean", "size"])
    if order:
        grouped = grouped.reindex([o for o in order if o in grouped.index])
    overall = df[cfg.TARGET].mean()
    fig = go.Figure(go.Bar(
        x=grouped.index.astype(str), y=grouped["mean"],
        marker=dict(color=t.ACCENT, cornerradius=4, line=dict(color=t.SURFACE, width=2)),
        customdata=grouped["size"],
        hovertemplate="%{x}<br>purchase rate %{y:.1%}<br>%{customdata:,} sessions<extra></extra>",
    ))
    fig.add_hline(y=overall, line=dict(color=t.INK_MUTED, dash="dot", width=1),
                  annotation_text=f"overall {overall:.1%}", annotation_position="top left",
                  annotation_font_color=t.INK_SECONDARY)
    fig.update_yaxes(tickformat=".0%", title="Purchase rate")
    fig.update_xaxes(type="category")
    return _base_layout(fig, height=300, showlegend=False)
