"""Colours and the custom CSS layer on top of the Streamlit theme.

Chart colours follow one rule: a model always keeps the same colour on every
chart (fixed registry order, never re-assigned by rank).
"""

import streamlit as st

from shopper_intent.models import MODELS

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
SURFACE = "#fcfcfb"
SURFACE_RAISED = "#ffffff"
GRID = "#e6e5e1"
ACCENT = "#2a78d6"

# categorical slots 1-5, validated colour-blind-safe for adjacent pairs
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MODEL_COLORS = {spec.name: color for spec, color in zip(MODELS.values(), _SERIES)}

# two-class charts: the class we care about gets the accent, the majority stays neutral
CLASS_COLORS = {"No purchase": "#a3a29c", "Purchase": ACCENT}

# one-hue blue ramp for magnitudes (confusion matrix)
SEQUENTIAL = ["#e8f1fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281"]

FONT_FAMILY = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {{ font-family: {FONT_FAMILY}; }}
.block-container {{ padding-top: 3.4rem; max-width: 1240px; }}

/* hero banner */
.hero {{
  border: 1px solid {GRID}; border-radius: 16px; padding: 1.4rem 1.6rem;
  background: linear-gradient(135deg, #f4f8fe 0%, {SURFACE_RAISED} 60%);
  margin-bottom: 1.2rem;
}}
.hero .eyebrow {{ color: {ACCENT}; font-weight: 600; font-size: .78rem;
  letter-spacing: .08em; text-transform: uppercase; }}
.hero h1 {{ font-size: 1.85rem; font-weight: 700; margin: .25rem 0 .35rem; color: {INK}; padding: 0; }}
.hero p {{ color: {INK_SECONDARY}; margin: 0; font-size: .98rem; max-width: 820px; }}
.hero .chips {{ margin-top: .9rem; display: flex; gap: .45rem; flex-wrap: wrap; }}
.chip {{ border: 1px solid {GRID}; background: {SURFACE_RAISED}; border-radius: 999px;
  padding: .18rem .7rem; font-size: .8rem; color: {INK_SECONDARY}; }}
.chip b {{ color: {INK}; font-weight: 600; }}

/* metric cards */
.metric-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: .7rem; margin: .4rem 0 1rem; }}
.metric-grid.compact {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
@media (max-width: 900px) {{ .metric-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
@media (max-width: 520px) {{ .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.metric-card {{ border: 1px solid {GRID}; border-radius: 12px; padding: .8rem .9rem;
  background: {SURFACE_RAISED}; }}
.metric-card .label {{ color: {INK_SECONDARY}; font-size: .78rem; font-weight: 500; }}
.metric-card .value {{ color: {INK}; font-size: 1.55rem; font-weight: 700;
  font-variant-numeric: tabular-nums; line-height: 1.25; }}
.metric-card .delta {{ font-size: .75rem; color: {INK_MUTED}; font-variant-numeric: tabular-nums; }}
.metric-card .delta.up {{ color: #1a7f4e; }}
.metric-card .delta.down {{ color: #c0392b; }}

/* section headings */
.section-title {{ font-size: 1.05rem; font-weight: 650; color: {INK}; margin: 1.2rem 0 .1rem; }}
.section-sub {{ font-size: .86rem; color: {INK_SECONDARY}; margin-bottom: .5rem; }}

/* callouts */
.callout {{ border-left: 3px solid {ACCENT}; background: #f4f8fe; padding: .7rem .9rem;
  border-radius: 0 10px 10px 0; color: {INK_SECONDARY}; font-size: .9rem; margin: .4rem 0 .8rem; }}
.callout b {{ color: {INK}; }}
.winner {{ border: 1px solid #cde2fb; background: #f4f8fe; border-radius: 12px;
  padding: .8rem 1rem; margin-bottom: .8rem; }}
.winner .name {{ font-weight: 700; font-size: 1.1rem; color: {INK}; }}
.winner .why {{ color: {INK_SECONDARY}; font-size: .88rem; }}
.swatch {{ display: inline-block; width: .7rem; height: .7rem; border-radius: 3px;
  margin-right: .4rem; vertical-align: baseline; }}

/* pipeline stepper */
.stepper {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: .6rem 0 .4rem; }}
.step {{ border: 1px solid {GRID}; background: {SURFACE_RAISED}; border-radius: 10px; padding: .4rem .7rem;
  font-size: .84rem; color: {INK}; }}
.step-dot {{ color: {ACCENT}; font-weight: 700; margin-right: .4rem; }}

/* sidebar */
section[data-testid="stSidebar"] .sidebar-brand {{ font-weight: 700; font-size: 1.05rem; color: {INK}; }}
section[data-testid="stSidebar"] .sidebar-caption {{ color: {INK_MUTED}; font-size: .78rem; }}

/* tabs: a little more breathing room */
.stTabs [data-baseweb="tab-list"] {{ gap: .3rem; }}
.stTabs [data-baseweb="tab"] {{ padding: .5rem .9rem; font-weight: 500; }}

code, .mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}
footer {{ visibility: hidden; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
