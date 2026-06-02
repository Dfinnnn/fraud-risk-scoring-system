"""
core/explainability.py

SHAP-based explainability for the CatBoost model.

Why CatBoost only:
- the supervised_score is CatBoost alone (DNN was dropped from the score
  after empirical fusion analysis), so explaining CatBoost explains the
  primary risk decision.
- DNN contributes no score weight; the AE outputs reconstruction error,
  not a probability, so SHAP does not apply to it meaningfully.

Scope:
- this module is UI-free. It exposes two functions:
    explain_one(row)     -> top-K SHAP features for a single transaction
    explain_batch(df)     -> top-K SHAP features for every row in a batch
- the Streamlit page decides which to call (single / batch CSV). The mode
  picker lives in the UI, not here.

Honest limitation:
- SHAP explains why CatBoost's fraud probability is high/low.
- It does NOT explain tier escalation (entity risk, anomaly, early warning).
  Those come from escalation logic and should be shown separately as the
  plain-text escalation_reason. Explanation = SHAP (model) + escalation_reason (logic).
"""

from typing import List, Union, Optional

import numpy as np
import pandas as pd

import utils.config as config
from utils.schema import ExplanationItem
from models.catboost_inference import CatBoostInferenceModel

# SHAP is imported lazily inside the explainer so that importing this module
# does not hard-require shap unless explanations are actually used.


class CatBoostExplainer:
    """
    Wraps a loaded CatBoost model with a SHAP TreeExplainer.

    Reuses CatBoostInferenceModel for loading and input preparation, so the
    feature alignment and categorical handling exactly match scoring.
    """

    def __init__(
        self,
        top_k: int = config.SHAP_TOP_K,
        cb_model: Optional[CatBoostInferenceModel] = None,
    ):
        self.top_k = top_k
        self._cb = cb_model or CatBoostInferenceModel()
        self._explainer = None
        self._loaded = False

    def load(self) -> None:
        """Load the CatBoost model and build the SHAP TreeExplainer."""
        import shap  # lazy import

        if self._cb._model is None:
            self._cb.load()

        # TreeExplainer is the correct, fast SHAP explainer for CatBoost.
        self._explainer = shap.TreeExplainer(self._cb._model)
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Explainer not loaded. Call CatBoostExplainer.load() first.")

    # ---------------------------------------------------------------
    # Internal: compute SHAP values for a prepared matrix
    # ---------------------------------------------------------------
    def _shap_values_for(self, df: pd.DataFrame) -> (np.ndarray, pd.DataFrame):
        """
        Prepare input the same way scoring does, then compute SHAP values.
        Returns (shap_values_2d, prepared_X).
        """
        from catboost import Pool

        X = self._cb._prepare_input(df)

        # Build a Pool so categorical features are handled correctly by SHAP.
        cat_idx = [
            X.columns.get_loc(c) for c in self._cb._cat_features if c in X.columns
        ]
        pool = Pool(X, cat_features=cat_idx)

        shap_values = self._explainer.shap_values(pool)
        # For binary CatBoost, shap_values is a 2D array (n_rows, n_features).
        shap_values = np.asarray(shap_values)

        return shap_values, X

    # ---------------------------------------------------------------
    # Internal: turn one row's SHAP vector into top-K ExplanationItems
    # ---------------------------------------------------------------
    def _top_k_items(
        self,
        shap_row: np.ndarray,
        feature_row: pd.Series,
    ) -> List[ExplanationItem]:
        feature_names = list(feature_row.index)

        # Rank features by absolute SHAP contribution (largest impact first).
        order = np.argsort(np.abs(shap_row))[::-1][: self.top_k]

        items: List[ExplanationItem] = []
        for idx in order:
            items.append(
                ExplanationItem(
                    feature_name=str(feature_names[idx]),
                    shap_value=float(shap_row[idx]),
                    feature_value=str(feature_row.iloc[idx]),
                )
            )
        return items

    # ---------------------------------------------------------------
    # Public: explain a single transaction
    # ---------------------------------------------------------------
    def explain_one(self, row: Union[pd.Series, dict]) -> List[ExplanationItem]:
        """
        Return the top-K SHAP feature contributions for one transaction.
        Positive shap_value pushes toward fraud; negative pushes away.
        """
        self._ensure_loaded()

        df = pd.DataFrame([row])
        shap_values, X = self._shap_values_for(df)

        return self._top_k_items(shap_values[0], X.iloc[0])

    # ---------------------------------------------------------------
    # Public: explain a batch
    # ---------------------------------------------------------------
    def explain_batch(self, df: pd.DataFrame) -> List[List[ExplanationItem]]:
        """
        Return a list (one entry per row) of top-K SHAP ExplanationItems.
        Intended for thesis/batch reporting on a CSV of transactions.
        """
        self._ensure_loaded()

        shap_values, X = self._shap_values_for(df)

        results: List[List[ExplanationItem]] = []
        for i in range(len(X)):
            results.append(self._top_k_items(shap_values[i], X.iloc[i]))
        return results


if __name__ == "__main__":
    # Sanity check: explain a few rows from the CatBoost val file.
    if not config.ENABLE_SHAP:
        print("[Explainability] ENABLE_SHAP is False in config. Nothing to do.")
    else:
        print("[Explainability] Loading CatBoost + SHAP explainer...")
        explainer = CatBoostExplainer()
        explainer.load()

        val = pd.read_csv(config.CATBOOST_VAL_FILE).head(3)
        print(f"[Explainability] Explaining {len(val)} sample rows, top_k={explainer.top_k}\n")

        batch = explainer.explain_batch(val)
        for i, items in enumerate(batch):
            print(f"--- Row {i} ---")
            for it in items:
                direction = "↑fraud" if it.shap_value > 0 else "↓fraud"
                print(f"  {it.feature_name} = {it.feature_value} | "
                      f"shap={it.shap_value:+.4f} ({direction})")
            print()