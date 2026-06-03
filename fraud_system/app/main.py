"""
app/main.py

Navigation entry point for FraudZilla.

Run from the project root (fraud_system/):
    streamlit run app/main.py

Requires Streamlit >= 1.36 (st.navigation / st.Page / Material icons).

This file owns three things and nothing else:
1. the single st.set_page_config call for the whole app,
2. the global theme + branded sidebar,
3. the page registry (order, titles, icons).

Each page's logic lives in its own file. Because this entry script reruns
before every page, apply_theme() here styles all pages, and the page files no
longer call st.set_page_config themselves (st.navigation requires it set once,
here).
"""

import state  # MUST be first: fixes sys.path
import ui

import streamlit as st


st.set_page_config(
    page_title="FraudZilla — Fraud Risk Scoring",
    page_icon="\U0001F6E1\uFE0F",  # browser-tab favicon only
    layout="wide",
)

ui.apply_theme()
ui.sidebar_brand()

# ---- Page registry (sidebar order + clean labels + Material icons) ----
overview = st.Page("overview.py", title="Overview", icon=":material/dashboard:", default=True)
scoring = st.Page("pages/1_Transaction_Scoring.py", title="Transaction Scoring", icon=":material/receipt_long:")
entity = st.Page("pages/3_Entity_Profile.py", title="Entity Profile", icon=":material/account_circle:")
monitor = st.Page("pages/4_System_Monitor.py", title="System Monitor", icon=":material/monitoring:")
explain = st.Page("pages/2_Explanation_Panel.py", title="Explanation Panel", icon=":material/insights:")

nav = st.navigation([overview, scoring, entity, monitor, explain])

ui.sidebar_status(state.pipeline_is_loaded())

nav.run()