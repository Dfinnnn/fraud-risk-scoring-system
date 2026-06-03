"""
app/pages/1_Transaction_Scoring.py

Page 1 — Transaction Scoring.

Pick a real, pre-engineered transaction from the sample data, score it through
the full pipeline, and show every decision field. Scoring here also updates the
session entity store, so the Entity Profile page reflects it.

Why a sample picker instead of a manual form:
the model needs all 168 engineered features. A manual form would leave ~160 of
them blank, the inference layer would silently fill them with 0 / "unknown",
and the resulting score would be meaningless. Picking a real engineered row
keeps the demonstrated score honest.
"""

import state  # MUST be first: fixes sys.path

import streamlit as st

import utils.config as config


st.set_page_config(page_title="Transaction Scoring", page_icon="🧾", layout="wide")
st.title("Transaction Scoring")

if not state.pipeline_is_loaded():
    st.warning("Models are not loaded. Go to the main page and click **Load models** first.")
    st.stop()

pipe = state.get_pipeline()

# ---------------------------------------------------------------
# Pick a transaction
# ---------------------------------------------------------------
st.subheader("1. Choose a transaction")

c1, c2 = st.columns([1, 2])
with c1:
    source = st.radio(
        "Sample source",
        options=["test", "val"],
        format_func=lambda s: "Test (unlabelled, blind input)" if s == "test"
        else "Validation (labelled)",
        help="Test mimics real unseen transactions. Validation rows have a known "
             "isFraud label if you want ground truth.",
    )

try:
    sample_df = state.load_sample_frame(source=source, n=500)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

id_col = config.TRANSACTION_ID_COL
ent_col = config.ENTITY_ID_COL

with c2:
    if id_col in sample_df.columns:
        options = sample_df[id_col].astype(str).tolist()
        chosen_id = st.selectbox(f"{id_col}", options)
        row_df = sample_df[sample_df[id_col].astype(str) == chosen_id].head(1)
    else:
        idx = st.number_input("Row index", 0, len(sample_df) - 1, 0)
        row_df = sample_df.iloc[[int(idx)]]
        chosen_id = str(idx)

row = row_df.iloc[0].to_dict()

# Quick context on the chosen row
info_cols = st.columns(4)
info_cols[0].metric("Entity (UID)", str(row.get(ent_col, "n/a")))
if "TransactionAmt" in row:
    info_cols[1].metric("Amount", f"{float(row['TransactionAmt']):,.2f}")
if config.TARGET_COL in row:
    info_cols[2].metric("True label", "Fraud" if int(row[config.TARGET_COL]) == 1 else "Legit")
info_cols[3].metric("Features in row", row_df.shape[1])

# ---------------------------------------------------------------
# Score
# ---------------------------------------------------------------
st.subheader("2. Score")

if st.button("Score transaction", type="primary"):
    result = pipe.score_transaction(row, explain=False)
    state.remember_scored_row(result.transaction_id, row, result)
    st.session_state["last_result"] = result

result = st.session_state.get("last_result")

if result is None:
    st.info("Pick a transaction and click **Score transaction**.")
    st.stop()

# ---------------------------------------------------------------
# Result display
# ---------------------------------------------------------------
st.subheader("3. Result")

tier = result.risk_tier or "Unknown"
tier_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(tier, "⚪")

top = st.columns(4)
top[0].metric("Transaction risk", f"{(result.transaction_risk or 0):.3f}")
top[1].metric("Risk tier", f"{tier_color} {tier}")
top[2].metric("Recommended action", result.recommended_action or "n/a")
top[3].metric("Anomaly flag", "Yes" if result.anomaly_flag else "No")

st.markdown("**Escalation reason**")
st.code(result.escalation_reason or "(none)", language=None)

with st.expander("Model signals (diagnostics)"):
    d = st.columns(2)
    d[0].metric("CatBoost probability (the score)", f"{(result.catboost_probability or 0):.4f}")
    d[1].metric("DNN probability (reported only)", f"{(result.dnn_probability or 0):.4f}")
    d2 = st.columns(2)
    d2[0].metric("Reconstruction error", f"{(result.reconstruction_error or 0):.4f}")
    d2[1].metric("Anomaly score", f"{(result.anomaly_score or 0):.4f}")
    st.caption(
        "The final risk uses CatBoost as the supervised score; the DNN value is "
        "shown for comparison only. Anomaly score feeds escalation, not the "
        "supervised score directly."
    )

# Entity context after this scoring
profile = pipe.entity_store.get(result.entity_id)
if profile is not None:
    with st.expander("Entity standing after this transaction"):
        e = st.columns(4)
        e[0].metric("Entity risk", f"{profile.entity_risk:.3f}")
        e[1].metric("Status", profile.entity_status)
        e[2].metric("Transactions seen", profile.transaction_count)
        e[3].metric("Early warning", "Active" if profile.early_warning_flag else "—")
        st.caption("Open the **Entity Profile** page to see the full risk history.")