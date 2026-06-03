"""
app/pages/4_System_Monitor.py

Page 4 — System Monitor / Batch Scoring.

Upload a CSV of pre-engineered transactions (or use the validation sample),
score the whole batch through the pipeline, then view summary statistics and
download the scored results.

Batch scoring sorts by TransactionDT internally so entity profiles build in
correct time order. Model inference is vectorised; the per-row decision loop
(fusion + entity update + escalation) is sequential, so very large files take
a while. Keep demo batches modest.

This page scores into the SAME session entity store, so a batch here also
shows up on the Entity Profile page.
"""

import state  # MUST be first: fixes sys.path

import io

import pandas as pd
import streamlit as st

import utils.config as config


st.title("System Monitor — Batch Scoring")

if not state.pipeline_is_loaded():
    st.warning("Models are not loaded. Open the **Overview** page and click **Initialize detection engine** first.")
    st.stop()

pipe = state.get_pipeline()

# ---------------------------------------------------------------
# 1. Choose input
# ---------------------------------------------------------------
st.subheader("1. Input")

src = st.radio(
    "Source",
    ["Upload CSV", "Validation sample"],
    horizontal=True,
    help="CSV must already be feature-engineered (same columns as the "
         "training data). Missing engineered features get filled with "
         "0/'unknown' and make scores unreliable.",
)

df = None
if src == "Upload CSV":
    up = st.file_uploader("Pre-engineered transactions CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
else:
    n = st.slider("Number of rows from validation set", 20, 2000, 200, step=20)
    df = state.load_sample_frame(source="val", n=n)

if df is None:
    st.info("Upload a CSV or switch to the validation sample.")
    st.stop()

st.write(f"Loaded **{len(df):,}** rows, **{df.shape[1]}** columns.")
st.dataframe(df.head(5), use_container_width=True)

# ---------------------------------------------------------------
# 2. Score
# ---------------------------------------------------------------
st.subheader("2. Score batch")

if len(df) > 1000:
    st.warning(
        f"{len(df):,} rows — the sequential decision loop will take a while. "
        "Consider a smaller batch for a live demo.",
        icon="⏳",
    )

if st.button("Score batch", type="primary"):
    with st.spinner(f"Scoring {len(df):,} transactions..."):
        results = pipe.score_batch(df, explain=False)
    # convert to a flat DataFrame for display / download
    rows = []
    for r in results:
        rows.append(
            {
                "transaction_id": r.transaction_id,
                "entity_id": r.entity_id,
                "transaction_risk": round(r.transaction_risk or 0, 4),
                "risk_tier": r.risk_tier,
                "recommended_action": r.recommended_action,
                "anomaly_flag": r.anomaly_flag,
                "catboost_probability": round(r.catboost_probability or 0, 4),
                "dnn_probability": round(r.dnn_probability or 0, 4),
                "anomaly_score": round(r.anomaly_score or 0, 4),
                "escalation_reason": r.escalation_reason,
            }
        )
    st.session_state["batch_results"] = pd.DataFrame(rows)

res_df = st.session_state.get("batch_results")
if res_df is None:
    st.info("Click **Score batch** to run the pipeline.")
    st.stop()

# ---------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------
st.subheader("3. Summary")

total = len(res_df)
flagged = int(res_df["anomaly_flag"].sum())
high = int((res_df["risk_tier"] == "High").sum())
blocked = int((res_df["recommended_action"] == "block").sum())

s = st.columns(4)
s[0].metric("Transactions", f"{total:,}")
s[1].metric("High tier", f"{high:,}", f"{high/total*100:.1f}%")
s[2].metric("Blocked", f"{blocked:,}", f"{blocked/total*100:.1f}%")
s[3].metric("Anomaly flagged", f"{flagged:,}", f"{flagged/total*100:.1f}%")

c1, c2 = st.columns(2)
with c1:
    st.caption("Risk tier distribution")
    tier_counts = (
        res_df["risk_tier"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .astype(int)
    )
    st.bar_chart(tier_counts)
with c2:
    st.caption("Recommended action distribution")
    action_counts = (
        res_df["recommended_action"]
        .value_counts()
        .reindex(["approve", "review", "block"])
        .fillna(0)
        .astype(int)
    )
    st.bar_chart(action_counts)

# ---------------------------------------------------------------
# 4. Results table + download
# ---------------------------------------------------------------
st.subheader("4. Results")

tier_filter = st.multiselect(
    "Filter by tier",
    ["Low", "Medium", "High"],
    default=["Low", "Medium", "High"],
)
view = res_df[res_df["risk_tier"].isin(tier_filter)]
st.dataframe(view, use_container_width=True, hide_index=True)

csv_buf = io.StringIO()
res_df.to_csv(csv_buf, index=False)
st.download_button(
    "Download full results CSV",
    data=csv_buf.getvalue(),
    file_name="batch_scoring_results.csv",
    mime="text/csv",
)

st.caption(
    f"Entities now tracked this session: {len(pipe.entity_store)}. "
    "Batch-scored entities also appear on the Entity Profile page."
)