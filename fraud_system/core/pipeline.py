"""
core/pipeline.py

End-to-end fraud scoring pipeline (Option 1: TransactionResult output).

Stage order per transaction:
    raw txn
      -> CatBoost probability
      -> DNN probability        (computed and reported, NOT blended into score)
      -> Autoencoder anomaly (reconstruction_error, anomaly_score, anomaly_flag)
      -> fusion: supervised_score (CatBoost only) + transaction_risk + risk_tier
      -> entity_profile.update()   (risk decay + counters)
      -> escalation.apply_escalation()  (rules + early warning, may upgrade tier)
      -> optional SHAP explanation (only when explain=True)
      -> TransactionResult

Design (confirmed):
- models load ONCE in __init__ and are reused.
- SHAP runs on demand only (explain=True), never by default.
- entity identity comes from config.ENTITY_ID_COL.
- batch scoring SORTS by TransactionDT so entity profiles build in
  correct time order; original transaction_id is preserved in output
  so results can be joined back to the source.
"""

from typing import List, Optional, Union

import pandas as pd

import utils.config as config
from utils.schema import TransactionResult

from models.catboost_inference import CatBoostInferenceModel
from models.dnn_inference import DNNInferenceModel
from models.autoencoder_inference import AutoencoderInferenceModel

from core import fusion
from core.entity_profile import EntityProfileStore
from core.escalation import apply_escalation


class FraudPipeline:
    """
    Loads all models once and scores transactions end to end.
    """

    def __init__(self, enable_shap: bool = False):
        # --- load supervised + anomaly models once ---
        self.catboost = CatBoostInferenceModel()
        self.catboost.load()

        self.dnn = DNNInferenceModel()
        self.dnn.load()

        self.autoencoder = AutoencoderInferenceModel()
        self.autoencoder.load()

        # --- entity profile store (in-memory, persists across calls) ---
        self.entity_store = EntityProfileStore()

        # --- optional SHAP explainer (loaded only if requested) ---
        self.enable_shap = enable_shap and config.ENABLE_SHAP
        self._explainer = None
        if self.enable_shap:
            from core.explainability import CatBoostExplainer
            self._explainer = CatBoostExplainer(cb_model=self.catboost)
            self._explainer.load()

        self.entity_id_col = config.ENTITY_ID_COL
        self.transaction_id_col = config.TRANSACTION_ID_COL
        self.time_col = "TransactionDT"

    # ---------------------------------------------------------------
    # Single transaction
    # ---------------------------------------------------------------
    def score_transaction(
        self,
        row: Union[pd.Series, dict],
        explain: bool = False,
    ) -> TransactionResult:
        """
        Score one transaction through the full pipeline.
        `explain=True` attaches SHAP top-K features (requires enable_shap).
        """
        df = pd.DataFrame([row])
        return self._score_df_rows(df, explain=explain)[0]

    # ---------------------------------------------------------------
    # Batch
    # ---------------------------------------------------------------
    def score_batch(
        self,
        df: pd.DataFrame,
        explain: bool = False,
    ) -> List[TransactionResult]:
        """
        Score many transactions. Sorts by TransactionDT first so entity
        profiles update in correct time order. Original transaction_id is
        preserved in each result for joining back to the source.
        """
        work = df.copy()
        if self.time_col in work.columns:
            work = work.sort_values(self.time_col).reset_index(drop=True)
        else:
            print(f"[Pipeline][WARN] '{self.time_col}' not found; "
                  f"scoring in given order. Entity time-ordering not guaranteed.")

        return self._score_df_rows(work, explain=explain)

    # ---------------------------------------------------------------
    # Core scoring loop
    # ---------------------------------------------------------------
    def _score_df_rows(
        self,
        df: pd.DataFrame,
        explain: bool,
    ) -> List[TransactionResult]:
        # Batch model inference (vectorised) for speed.
        cb_df = self.catboost.predict_scores(df)        # transaction_id, entity_id, catboost_probability
        dnn_df = self.dnn.predict_scores(df)            # transaction_id, entity_id, dnn_probability
        ae_df = self.autoencoder.score_df(df)     # transaction_id, entity_id, reconstruction_error, anomaly_score, anomaly_flag

        # Optional SHAP for the whole batch at once.
        shap_batch = None
        if explain:
            if not self.enable_shap or self._explainer is None:
                print("[Pipeline][WARN] explain=True but SHAP not enabled; skipping explanations.")
            else:
                shap_batch = self._explainer.explain_batch(df)

        # Timestamps for entity profile (if present).
        timestamps = (
            df[self.time_col].values if self.time_col in df.columns else [None] * len(df)
        )

        results: List[TransactionResult] = []

        for i in range(len(df)):
            txn_id = str(cb_df.iloc[i]["transaction_id"])
            entity_id = str(cb_df.iloc[i]["entity_id"])

            cb_prob = float(cb_df.iloc[i]["catboost_probability"])
            dnn_prob = float(dnn_df.iloc[i]["dnn_probability"])

            recon_err = float(ae_df.iloc[i]["reconstruction_error"])
            anomaly_score = float(ae_df.iloc[i]["anomaly_score"])
            anomaly_flag = bool(ae_df.iloc[i]["anomaly_flag"])

            # --- fusion: supervised (CatBoost only) + transaction risk ---
            fused = fusion.fuse(
                catboost_probability=cb_prob,
                dnn_probability=dnn_prob,      # accepted, not blended
                anomaly_score=anomaly_score,
            )
            base_tier = fusion.assign_risk_tier(fused.transaction_risk)

            # --- entity profile update (uses transaction_risk) ---
            profile = self.entity_store.update(
                entity_id=entity_id,
                transaction_risk=fused.transaction_risk,
                risk_tier=base_tier,
                anomaly_flag=anomaly_flag,
                timestamp=timestamps[i],
            )

            # --- escalation (may upgrade tier, sets profile escalation fields) ---
            esc = apply_escalation(
                base_tier=base_tier,
                profile=profile,
                anomaly_flag=anomaly_flag,
            )

            # --- assemble TransactionResult ---
            result = TransactionResult(
                transaction_id=txn_id,
                entity_id=entity_id,
                catboost_probability=cb_prob,
                dnn_probability=dnn_prob,
                reconstruction_error=recon_err,
                anomaly_score=anomaly_score,
                anomaly_flag=anomaly_flag,
                supervised_score=fused.supervised_score,
                transaction_risk=fused.transaction_risk,
                risk_tier=esc["final_tier"],
                recommended_action=esc["recommended_action"],
                escalation_reason=esc["escalation_reason"],
                model_version=config.MODEL_VERSION if hasattr(config, "MODEL_VERSION") else None,
            )

            if shap_batch is not None:
                result.top_shap_features = shap_batch[i]

            results.append(result)

        return results


if __name__ == "__main__":
    print("[Pipeline] Initialising (loading all models)...")
    pipe = FraudPipeline(enable_shap=False)
    print("[Pipeline] Ready.\n")

    # Score a small batch from the CatBoost val file as a smoke test.
    sample = pd.read_csv(config.CATBOOST_VAL_FILE).head(10)
    results = pipe.score_batch(sample, explain=False)

    print(f"Scored {len(results)} transactions.\n")
    for r in results[:10]:
        print(
            f"{r.transaction_id} | risk={r.transaction_risk:.3f} | "
            f"tier={r.risk_tier} | action={r.recommended_action} | "
            f"anomaly={r.anomaly_flag} | reason={r.escalation_reason}"
        )

    print(f"\nEntities tracked: {len(pipe.entity_store)}")