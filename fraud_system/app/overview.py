"""
app/overview.py

Overview page — two states on one page:

1. Not loaded  -> a focused "load the engine" gate (centered card + button).
2. Loaded      -> status bar + workspace cards + how-to-read-the-scores scope.

Navigation between workspaces is handled by the branded sidebar (defined in
main.py), so the cards here are informational rather than clickable. This keeps
the page stable: no fragile custom click targets, just a clean state swap.

Note: the gate does not lock the other pages — Streamlit always lists them in
the sidebar. Each page has its own "models not loaded" guard as the safety net.
"""

import state
import ui

import streamlit as st

import utils.config as config


loaded = state.pipeline_is_loaded()

# ---------------------------------------------------------------
# State 1 — load gate
# ---------------------------------------------------------------
if not loaded:
    st.markdown(
        f"""
        <div class="fz-gate">
          <div class="logo">{ui.icon('shield')}</div>
          <div class="eyebrow">Fraud Detection · Prototype</div>
          <h1>Power up FraudZilla</h1>
          <p>Load the scoring models into this session to start detecting fraud.
             Models load once and stay in memory while the app runs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c = st.columns([1, 1, 1])
    with c[1]:
        st.write("")
        if st.button("Initialize detection engine", type="primary", use_container_width=True):
            state.get_pipeline(enable_shap=False)
            st.rerun()
    st.stop()

# ---------------------------------------------------------------
# State 2 — overview
# ---------------------------------------------------------------
pipe = state.get_pipeline()

ui.status_bar(
    loaded=True,
    version=str(getattr(config, "MODEL_VERSION", "n/a")),
    entity_key=config.ENTITY_ID_COL,
    n_entities=len(pipe.entity_store),
)

ui.hero(
    eyebrow="Fraud Detection · Prototype",
    title="FraudZilla",
    subtitle_html=(
        "A hybrid risk-scoring engine for transaction fraud: a <strong>CatBoost</strong> "
        "scorer backed by an autoencoder anomaly signal, dynamic per-entity risk, and "
        "rule-based escalation — every decision traceable with <strong>SHAP</strong>."
    ),
)

ui.section("Start here", hint="Four workspaces · open any from the sidebar")
ui.start_cards()

ui.scope_list(
    [
        "The final risk is the <b>CatBoost probability</b>; the DNN value is reported for comparison only and is not blended in.",
        "The <b>autoencoder</b> provides an auxiliary anomaly signal used by escalation — not a co-equal fraud probability.",
        "<b>SHAP</b> explains the CatBoost probability, not the tier escalation, which comes from the rules engine.",
        "Inputs must already be <b>feature-engineered</b> (168 features); use the built-in sample picker for trustworthy scores.",
    ]
)