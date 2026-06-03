"""
app/main.py

Landing page for the Hybrid Fraud Detection & Risk Scoring System.

Run from the project root (fraud_system/):
    streamlit run app/main.py

Streamlit auto-discovers the other pages in app/pages/ and lists them in the
sidebar. This page loads the models once and explains what the system does.
"""

import state  # MUST be first: fixes sys.path and exposes shared helpers

import streamlit as st

import utils.config as config


st.set_page_config(
    page_title="Fraud Risk Scoring System",
    page_icon="🛡️",
    layout="wide",
)

st.title("Hybrid Fraud Detection & Dynamic Risk Scoring")
st.caption(
    "CatBoost primary scorer · Autoencoder anomaly signal · "
    "dynamic entity risk · rule-based escalation · SHAP explainability"
)

# ---------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("System status")
    if state.pipeline_is_loaded():
        st.success("Models loaded and ready.")
    else:
        st.info("Models not loaded yet.")
        if st.button("Load models", type="primary"):
            state.get_pipeline(enable_shap=True)
            st.rerun()

    if state.pipeline_is_loaded():
        pipe = state.get_pipeline()
        st.metric("Entities tracked this session", len(pipe.entity_store))
        st.write(f"Model version: `{getattr(config, 'MODEL_VERSION', 'n/a')}`")
        st.write(f"Entity key: `{config.ENTITY_ID_COL}`")

with right:
    st.subheader("What each page does")
    st.markdown(
        "- **Transaction Scoring** — score one transaction, see tier, action, "
        "anomaly flag and escalation reason.\n"
        "- **Entity Profile** — risk history and standing status for one entity.\n"
        "- **System Monitor** — batch-score a CSV, view summary and download.\n"
        "- **Explanation Panel** — SHAP top-5 drivers for a transaction."
    )

st.divider()

st.subheader("Honest scope")
st.markdown(
    "- The final risk is **CatBoost only**; the DNN is reported as a diagnostic "
    "but is not blended into the score.\n"
    "- The **Autoencoder** contributes an auxiliary anomaly signal used by "
    "escalation, not a co-equal fraud probability.\n"
    "- SHAP explains the **CatBoost probability**, not the tier escalation "
    "(entity risk / anomaly / early warning) — those are shown as plain text.\n"
    "- Input must be **already feature-engineered** (168 features). The app does "
    "not run feature engineering; use the built-in sample picker for real scores."
)

st.info(
    "Entity profiles accumulate in memory for this browser session. They build "
    "up as you score transactions and persist across page switches, but reset "
    "if you restart the app.",
    icon="ℹ️",
)