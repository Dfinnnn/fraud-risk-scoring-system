"""
app/state.py

Shared application state for the Streamlit UI.

Responsibilities:
- put the project root on sys.path so `utils.*`, `core.*`, `models.*` import
  correctly when Streamlit runs scripts from the app/ directory,
- load the FraudPipeline EXACTLY ONCE and keep it in st.session_state
  (this is the EP-8 fix: the in-memory EntityProfileStore now survives
  page switches and reruns for the whole browser session),
- provide small helpers the pages reuse (sample loading, demo preload).

Every page must `import state` BEFORE importing any project module, because
importing this file is what fixes sys.path.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------
# Path bootstrap (must run before importing project modules)
# ---------------------------------------------------------------
# app/state.py  -> parents[1] is the fraud_system/ project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

import utils.config as config
from core.pipeline import FraudPipeline


# ---------------------------------------------------------------
# Pipeline (loaded once, kept for the whole session)
# ---------------------------------------------------------------
def get_pipeline(enable_shap: bool = False) -> FraudPipeline:
    """
    Return the single FraudPipeline for this session.

    The pipeline (and its in-memory entity store) is created once and stored
    in st.session_state, so entity profiles accumulate across transactions
    and survive page navigation. Models are heavy to load, so this also keeps
    the app fast after the first load.

    If a pipeline already exists but SHAP is now requested and was not enabled
    before, it is rebuilt once with SHAP on.
    """
    existing = st.session_state.get("pipeline")

    if existing is not None:
        if enable_shap and not getattr(existing, "enable_shap", False):
            # need SHAP now but the cached pipeline was built without it
            st.session_state.pop("pipeline", None)
        else:
            return existing

    with st.spinner("Loading models (first time only)..."):
        pipeline = FraudPipeline(enable_shap=enable_shap)

    st.session_state["pipeline"] = pipeline
    return pipeline


def pipeline_is_loaded() -> bool:
    return st.session_state.get("pipeline") is not None


# ---------------------------------------------------------------
# Scored-transaction history (for the Explanation page picker)
# ---------------------------------------------------------------
def remember_scored_row(transaction_id: str, row: dict, result) -> None:
    """
    Keep the raw input row + result for transactions scored on Page 1, so the
    Explanation page can re-run SHAP on a transaction the user already scored.
    """
    history = st.session_state.setdefault("scored_history", {})
    history[str(transaction_id)] = {"row": row, "result": result}


def get_scored_history() -> dict:
    return st.session_state.get("scored_history", {})


# ---------------------------------------------------------------
# Sample data loading
# ---------------------------------------------------------------
TEST_FILE = config.FEATURE_DIR / "test_catboost_ready.csv"
VAL_FILE = config.CATBOOST_VAL_FILE


@st.cache_data(show_spinner=False)
def load_sample_frame(source: str = "test", n: int = 500) -> pd.DataFrame:
    """
    Load a small slice of pre-engineered data for the sample picker.

    source="test" -> unlabelled, realistic blind input (test_catboost_ready.csv)
    source="val"  -> labelled data (val_catboost_ready.csv), useful when you
                     want a row whose true isFraud is known.

    Only the first `n` rows are read for responsiveness. These files are
    already feature-engineered (168 features), so scores are trustworthy.
    """
    path = TEST_FILE if source == "test" else VAL_FILE
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Sample file not found: {path}\n"
            f"Expected a pre-engineered CSV in {config.FEATURE_DIR}."
        )
    return pd.read_csv(path, nrows=n)


# ---------------------------------------------------------------
# Demo entity preload (run after we pick a demo entity)
# ---------------------------------------------------------------
def preload_entity(entity_id: str, source: str = "val", max_rows: int = 60) -> int:
    """
    Score every transaction belonging to one entity, in time order, through
    the session pipeline. This builds a meaningful entity profile (risk
    history, counters, early-warning flag) so Page 3 is not flat.

    Returns the number of transactions scored. Safe to call once per entity;
    re-calling will re-score and further inflate counters, so pages should
    guard with a "already preloaded" flag.

    NOTE: pick a demo entity that has several transactions but is NOT the
    99%-anomaly entity (15775_481.0_330.0_unknown) — that one floods anomaly
    flags and is a poor demo.
    """
    pipe = get_pipeline()
    df = pd.read_csv(VAL_FILE) if source == "val" else pd.read_csv(TEST_FILE)

    ent_col = config.ENTITY_ID_COL
    if ent_col not in df.columns:
        raise ValueError(f"Entity column '{ent_col}' not in {source} file.")

    subset = df[df[ent_col].astype(str) == str(entity_id)]
    if len(subset) == 0:
        return 0

    subset = subset.head(max_rows)
    pipe.score_batch(subset, explain=False)  # batch sorts by TransactionDT
    return len(subset)