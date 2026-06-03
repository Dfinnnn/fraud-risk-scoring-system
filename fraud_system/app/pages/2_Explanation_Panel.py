"""
app/pages/2_Explanation_Panel.py

Page 2 — Explanation Panel (SHAP).

Shows WHY CatBoost assigned its fraud probability, using SHAP top-K feature
contributions. Two modes:

- Single: explain one transaction (from a sample row, or a transaction you
  already scored on Page 1). Fast — runs TreeExplainer on one row.
- Batch: upload a CSV and explain every row. SLOW — TreeExplainer runs per row;
  warn before large files.

Honest scope (important): SHAP explains the CatBoost probability ONLY. It does
NOT explain tier escalation (entity risk, anomaly flag, early warning). Those
come from the rules engine and are shown separately as the plain-text
escalation_reason. Do not read the SHAP chart as an explanation of the tier.

This page needs SHAP enabled, so it requests a SHAP-enabled pipeline. If the
models were first loaded without SHAP, the pipeline is rebuilt once with it on.
"""

import state  # MUST be first: fixes sys.path

import io

import altair as alt
import pandas as pd
import streamlit as st

import utils.config as config


st.title("Explanation Panel — SHAP")

if not state.pipeline_is_loaded():
    st.warning("Models are not loaded. Open the **Overview** page and click **Initialize detection engine** first.")
    st.stop()

# Request a SHAP-enabled pipeline (rebuilds once if needed).
pipe = state.get_pipeline(enable_shap=True)
if not getattr(pipe, "enable_shap", False) or pipe._explainer is None:
    st.error(
        "SHAP is not available on the pipeline. Check that ENABLE_SHAP is True "
        "in config and the `shap` package is installed."
    )
    st.stop()

explainer = pipe._explainer

st.info(
    "SHAP explains the **CatBoost fraud probability** only — not the tier "
    "escalation. The escalation reason (entity risk / anomaly / early warning) "
    "is shown separately and comes from the rules engine.",
    icon="ℹ️",
)


def render_items(items):
    """Turn a list of ExplanationItem into a sorted SHAP bar chart + table."""
    if not items:
        st.info("No SHAP features returned.")
        return
    data = pd.DataFrame(
        {
            "feature": [it.feature_name for it in items],
            "shap_value": [it.shap_value for it in items],
            "feature_value": [it.feature_value for it in items],
            "direction": [
                "Toward fraud" if it.shap_value > 0 else "Away from fraud"
                for it in items
            ],
        }
    )

    # Order features by absolute impact (SHAP convention: biggest driver first),
    # regardless of direction. Color still encodes sign.
    order = (
        data.assign(_abs=data["shap_value"].abs())
        .sort_values("_abs", ascending=False)["feature"]
        .tolist()
    )

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadius=3, height={"band": 0.7})
        .encode(
            x=alt.X("shap_value:Q", title="SHAP contribution"),
            y=alt.Y(
                "feature:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelLimit=260, labelOverlap=False),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Toward fraud", "Away from fraud"],
                    range=["#B42318", "#216E4E"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["feature", "shap_value", "feature_value", "direction"],
        )
        .properties(height=54 * len(data) + 30)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Red pushes the score toward fraud; green pushes it away.")
    st.dataframe(data, use_container_width=True, hide_index=True)


mode = st.radio("Mode", ["Single transaction", "Batch CSV"], horizontal=True)

# ===============================================================
# SINGLE
# ===============================================================
if mode == "Single transaction":
    st.subheader("Single transaction")

    history = state.get_scored_history()
    sub = st.radio(
        "Pick from",
        ["A transaction I scored on Page 1", "A sample row"],
        horizontal=True,
    )

    row = None
    if sub == "A transaction I scored on Page 1":
        if not history:
            st.info("No scored transactions yet. Score one on Page 1, or use a sample row.")
        else:
            tid = st.selectbox("Scored transaction", list(history.keys()))
            row = history[tid]["row"]
            res = history[tid]["result"]
            d = st.columns(3)
            d[0].metric("CatBoost prob", f"{(res.catboost_probability or 0):.4f}")
            d[1].metric("Tier", res.risk_tier)
            d[2].metric("Action", res.recommended_action)
            st.caption(f"Escalation reason (rules, not SHAP): {res.escalation_reason}")
    else:
        sample_df = state.load_sample_frame(source="val", n=500)
        id_col = config.TRANSACTION_ID_COL
        if id_col in sample_df.columns:
            tid = st.selectbox(id_col, sample_df[id_col].astype(str).tolist())
            row = sample_df[sample_df[id_col].astype(str) == tid].head(1).iloc[0].to_dict()
        else:
            idx = st.number_input("Row index", 0, len(sample_df) - 1, 0)
            row = sample_df.iloc[int(idx)].to_dict()

    if row is not None and st.button("Explain", type="primary"):
        with st.spinner("Computing SHAP..."):
            items = explainer.explain_one(row)
        st.subheader("Top SHAP drivers")
        render_items(items)

# ===============================================================
# BATCH
# ===============================================================
else:
    st.subheader("Batch CSV")
    st.warning(
        "Batch SHAP runs the explainer on every row and is noticeably slow. "
        "Keep the file small for a demo (tens of rows, not thousands).",
        icon="⏳",
    )

    up = st.file_uploader("Pre-engineered transactions CSV", type=["csv"])
    if up is not None:
        bdf = pd.read_csv(up)
        st.write(f"Loaded **{len(bdf):,}** rows.")
        cap = st.number_input("Rows to explain (from top)", 1, min(len(bdf), 200), min(len(bdf), 20))

        if st.button("Explain batch", type="primary"):
            work = bdf.head(int(cap))
            with st.spinner(f"Computing SHAP for {len(work)} rows..."):
                batch_items = explainer.explain_batch(work)

            id_col = config.TRANSACTION_ID_COL
            ids = (
                work[id_col].astype(str).tolist()
                if id_col in work.columns
                else [str(i) for i in range(len(work))]
            )

            flat = []
            for tid, items in zip(ids, batch_items):
                for rank, it in enumerate(items, start=1):
                    flat.append(
                        {
                            "transaction_id": tid,
                            "rank": rank,
                            "feature": it.feature_name,
                            "shap_value": round(it.shap_value, 6),
                            "feature_value": it.feature_value,
                            "direction": "toward fraud" if it.shap_value > 0 else "away from fraud",
                        }
                    )
            out = pd.DataFrame(flat)
            st.dataframe(out, use_container_width=True, hide_index=True)

            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "Download SHAP explanations CSV",
                data=buf.getvalue(),
                file_name="shap_explanations.csv",
                mime="text/csv",
            )