"""Shopper Intent Lab - Streamlit front-end for BITS ML Assignment 2.

    streamlit run app.py

The sidebar picks the data (bundled test_data.csv or an uploaded csv) and the
model; the tabs guide the reviewer, evaluate, compare, trace the pipeline, score
single sessions and retrain.
Saved models in model/ are only ever loaded here, never refitted on the upload.
"""

import streamlit as st

from shopper_intent.data import SchemaError
from shopper_intent.models import MODELS
from ui import components as ui
from ui.state import ActiveData, bundled_test_bytes, get_metadata, load_active
from ui.theme import inject_css
from ui.views import about, compare, evaluate, guide, pipeline, predict, train_lab

BUNDLED = "Bundled test set"
UPLOAD = "Upload my own CSV"


def sidebar() -> tuple[ActiveData | None, str]:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🛒 Shopper Intent Lab</div>'
                    '<div class="sidebar-caption">2026AJ05820 · ML Assignment 2</div>',
                    unsafe_allow_html=True)
        st.divider()

        st.markdown("**1 · Data**")
        source = st.radio("Data source", [BUNDLED, UPLOAD], label_visibility="collapsed", key="source")
        data, error = None, None
        if source == UPLOAD:
            file = st.file_uploader(
                "Test CSV with the 17 feature columns (+ Revenue for metrics)", type=["csv"],
                help="Same columns as the UCI file. Upload test data only - models are not retrained.",
            )
            if file is not None:
                try:
                    data = load_active(file.name, file.getvalue())
                except SchemaError as exc:
                    error = str(exc)
        else:
            data = load_active("test_data.csv", bundled_test_bytes())

        st.download_button("Download sample test_data.csv", bundled_test_bytes(),
                           file_name="test_data.csv", mime="text/csv", width="stretch")
        if data is not None:
            target_note = "with labels" if data.labelled else "no Revenue column"
            st.success(f"**{data.name}** · {len(data.X):,} rows · {target_note}", icon="✅")
        if error:
            st.error(f"Can't use this file: {error}", icon="⚠️")

        st.markdown("**2 · Model**")
        model_key = st.selectbox("Model", list(MODELS), format_func=lambda k: MODELS[k].name, key="model",
                                 label_visibility="collapsed")
        st.caption(MODELS[model_key].blurb)

        meta = get_metadata()
        if meta:
            st.divider()
            st.caption(f"Models trained {meta['trained_at'][:10]} on {meta['n_train']:,} sessions · "
                       f"scikit-learn {meta['sklearn']}")
    return data, model_key


def main() -> None:
    st.set_page_config(page_title="Shopper Intent Lab · ML Assignment 2", page_icon="🛒",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()
    data, model_key = sidebar()

    meta = get_metadata()
    ui.hero(
        "Online Shoppers Purchasing Intention · UCI",
        "Will this visitor buy?",
        "Five classifiers trained on 12k e-commerce browsing sessions. Upload a test file, pick a model, "
        "and inspect how well each one finds the ~16% of sessions that end in a purchase.",
        chips={"Features": "17", "Train sessions": f"{meta.get('n_train', 0):,}",
               "Test sessions": f"{meta.get('n_test', 0):,}", "Models": str(len(MODELS))},
    )

    tab_guide, tab_eval, tab_compare, tab_pipeline, tab_predict, tab_lab, tab_about = st.tabs(
        ["👋 Start here", "📊 Evaluate", "🏁 Compare models", "⚙️ Pipeline", "🧪 Predict a session",
         "🛠️ Training lab", "ℹ️ About"]
    )
    with tab_guide:
        guide.render()
    with tab_eval:
        if data is None:
            st.info("Upload a CSV in the sidebar, or switch back to the bundled test set, to see results.")
        else:
            evaluate.render(data, model_key)
    with tab_compare:
        if data is None:
            st.info("Upload a CSV in the sidebar, or switch back to the bundled test set, to compare models.")
        else:
            compare.render(data)
    with tab_pipeline:
        if data is None:
            st.info("Upload a CSV in the sidebar, or switch back to the bundled test set, to run the pipeline.")
        else:
            pipeline.render(data, model_key)
    with tab_predict:
        predict.render()
    with tab_lab:
        # no upload yet -> the lab still works, scoring on the bundled test set
        train_lab.render(data or load_active("test_data.csv", bundled_test_bytes()))
    with tab_about:
        about.render()


main()
