"""
app/pages/3_Entity_Profile.py

Page 3 — Entity Profile Dashboard.

Shows the dynamic standing of one entity (UID): its decayed entity_risk,
status, counters, sticky early-warning flag, and the risk history over the
transactions seen this session.

Two ways to populate an entity:
1. Score transactions on Page 1 (they accumulate in the session entity store).
2. Use the demo preload below to score every transaction of one entity from
   the validation set at once — this is what makes the risk history chart
   meaningful instead of a single flat point.

Note on the decay (beta=0.85): entity_risk moves slowly on purpose. Even two
High transactions only lift it modestly; the escalation layer relies on
counters and trend/early-warning patterns, not entity_risk alone.
"""

import state  # MUST be first: fixes sys.path

import pandas as pd
import streamlit as st

import utils.config as config


st.title("Entity Profile Dashboard")

if not state.pipeline_is_loaded():
    st.warning("Models are not loaded. Open the **Overview** page and click **Initialize detection engine** first.")
    st.stop()

pipe = state.get_pipeline()
store = pipe.entity_store

# ---------------------------------------------------------------
# Demo preload (build a meaningful history for one entity)
# ---------------------------------------------------------------
st.subheader("Populate an entity")

preloaded = st.session_state.setdefault("preloaded_entities", set())

with st.expander("Demo preload — score all transactions of one entity", expanded=(len(store) == 0)):
    st.caption(
        "Scores every transaction for the given entity (from the validation "
        "set, in time order). Pick an entity with several transactions to see "
        "the risk history evolve. Avoid the 99%-anomaly entity "
        "`15775_481.0_330.0_unknown` — it floods anomaly flags."
    )
    colp = st.columns([3, 1, 1])
    with colp[0]:
        demo_id = st.text_input(
            f"{config.ENTITY_ID_COL} to preload",
            value="",
            placeholder="e.g. 9633_130.0_299.0_hotmail.com",
        )
    with colp[1]:
        max_rows = st.number_input("Max txns", 5, 500, 60, step=5)
    with colp[2]:
        st.write("")
        st.write("")
        do_preload = st.button("Preload", type="primary")

    if do_preload and demo_id.strip():
        key = demo_id.strip()
        if key in preloaded:
            st.info("Already preloaded this entity this session (skipped to avoid double-counting).")
        else:
            try:
                n = state.preload_entity(key, source="val", max_rows=int(max_rows))
                if n == 0:
                    st.error(f"No transactions found for `{key}` in the validation set.")
                else:
                    preloaded.add(key)
                    st.success(f"Scored {n} transactions for `{key}`.")
            except Exception as e:
                st.error(f"Preload failed: {e}")

# ---------------------------------------------------------------
# Select an entity to view
# ---------------------------------------------------------------
st.subheader("View an entity")

if len(store) == 0:
    st.info("No entities tracked yet. Preload one above, or score transactions on Page 1.")
    st.stop()

profiles = store.all_profiles()
# Sort by transaction count desc so the richest profiles are easy to find.
ordered_ids = sorted(profiles.keys(), key=lambda k: profiles[k].transaction_count, reverse=True)

chosen = st.selectbox(
    "Tracked entities (this session)",
    ordered_ids,
    format_func=lambda k: f"{k}  ({profiles[k].transaction_count} txns)",
)

profile = store.get(chosen)
if profile is None:
    st.error("Entity not found.")
    st.stop()

# ---------------------------------------------------------------
# Standing summary
# ---------------------------------------------------------------
status_icon = {"Clean": "🟢", "Watch": "🟡", "Risky": "🔴"}.get(profile.entity_status, "⚪")

m = st.columns(4)
m[0].metric("Entity risk", f"{profile.entity_risk:.3f}")
m[1].metric("Status", f"{status_icon} {profile.entity_status}")
m[2].metric("Transactions seen", profile.transaction_count)
m[3].metric("Early warning", "🚨 Active" if profile.early_warning_flag else "—")

m2 = st.columns(4)
m2[0].metric("High-risk txns", profile.high_risk_count)
m2[1].metric("Anomaly flags", profile.anomaly_count)
m2[2].metric("Reviews", getattr(profile, "review_count", 0))
m2[3].metric("Blocks", getattr(profile, "block_count", 0))

if profile.early_warning_flag:
    st.warning(
        "Early-warning flag is active and sticky — it stays on once triggered, "
        "even if later transactions look clean. This is monitoring-only in the "
        "current build (it does not by itself force a review action).",
        icon="🚨",
    )

# ---------------------------------------------------------------
# Risk history (the key visual)
# ---------------------------------------------------------------
st.subheader("Risk history")

history = list(profile.risk_history or [])
if len(history) == 0:
    st.info("No risk history recorded for this entity.")
elif len(history) == 1:
    st.info(
        f"Only one transaction recorded (risk {history[0]:.3f}). The risk "
        "history chart needs at least two transactions to be meaningful — "
        "use the demo preload above to score several transactions for this entity."
    )
else:
    anomalies = list(getattr(profile, "anomaly_history", []) or [])
    # pad anomaly list if shorter, just in case
    if len(anomalies) < len(history):
        anomalies = anomalies + [False] * (len(history) - len(anomalies))

    chart_df = pd.DataFrame(
        {
            "transaction_risk": history,
            "Medium threshold": [config.RISK_TIER_LOW_MAX] * len(history),
            "High threshold": [config.RISK_TIER_MEDIUM_MAX] * len(history),
        },
        index=pd.RangeIndex(1, len(history) + 1, name="transaction #"),
    )
    st.line_chart(chart_df)

    flagged = sum(1 for a in anomalies if a)
    cc = st.columns(3)
    cc[0].metric("Latest risk", f"{history[-1]:.3f}")
    cc[1].metric("Peak risk", f"{max(history):.3f}")
    cc[2].metric("Anomaly rate", f"{(flagged / len(anomalies) * 100 if anomalies else 0):.0f}%")

    with st.expander("Per-transaction detail"):
        detail = pd.DataFrame(
            {
                "transaction #": range(1, len(history) + 1),
                "transaction_risk": [round(r, 4) for r in history],
                "anomaly_flag": anomalies,
            }
        )
        st.dataframe(detail, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# Timestamps / last action
# ---------------------------------------------------------------
with st.expander("Other fields"):
    f = st.columns(3)
    f[0].write(f"**First seen (DT):** {profile.first_seen}")
    f[1].write(f"**Last seen (DT):** {profile.last_seen}")
    f[2].write(f"**Last action:** {getattr(profile, 'last_action', None)}")
    st.write(f"**Risk trend flag (live):** {profile.risk_trend_flag}")