"""
core/disagreement_analysis.py

Purpose:
- decide the DNN's real role by measuring complementary coverage,
  not by blended AUC-PR.

Question this answers:
- Does the DNN catch fraud cases that CatBoost misses (and vice versa)?
- If yes -> DNN has value as a parallel/escalation signal.
- If no  -> DNN is redundant for scoring and should be dropped from the blend.

Method:
- load val predictions for CatBoost and DNN.
- apply each model's own decision threshold to get caught/missed fraud.
- compare the fraud sets:
    * caught by both
    * caught by CatBoost only
    * caught by DNN only
    * missed by both
- also report a threshold-independent view: among frauds CatBoost ranks
  LOW (would miss), how does the DNN rank them?

Note on thresholds:
- CatBoost default threshold 0.50 is used as a baseline.
- DNN at 0.50 predicted almost nothing earlier, so we use a DNN
  operating threshold that reflects its own score distribution.
  We report results at a few DNN thresholds so the picture is honest,
  not cherry-picked.
"""

import numpy as np
import pandas as pd

import utils.config as config


def load_merged() -> pd.DataFrame:
    cb = pd.read_csv(config.CATBOOST_VAL_PREDICTIONS_FILE)
    dnn = pd.read_csv(config.DNN_VAL_PREDICTIONS_FILE)

    cb = cb[["transaction_id", "catboost_probability", "y_true"]]
    dnn = dnn[["transaction_id", "dnn_probability"]]

    merged = cb.merge(dnn, on="transaction_id", how="inner")
    return merged


def coverage_at_thresholds(
    df: pd.DataFrame,
    cb_threshold: float,
    dnn_threshold: float,
) -> dict:
    """
    Compare fraud coverage of the two models at given thresholds.
    Only fraud rows (y_true == 1) are analysed for catch/miss overlap.
    """
    fraud = df[df["y_true"] == 1]
    total_fraud = len(fraud)

    cb_caught = fraud["catboost_probability"] >= cb_threshold
    dnn_caught = fraud["dnn_probability"] >= dnn_threshold

    both = (cb_caught & dnn_caught).sum()
    cb_only = (cb_caught & ~dnn_caught).sum()
    dnn_only = (~cb_caught & dnn_caught).sum()
    neither = (~cb_caught & ~dnn_caught).sum()

    return {
        "cb_threshold": cb_threshold,
        "dnn_threshold": dnn_threshold,
        "total_fraud": total_fraud,
        "caught_by_both": int(both),
        "caught_by_catboost_only": int(cb_only),
        "caught_by_dnn_only": int(dnn_only),
        "missed_by_both": int(neither),
        "dnn_unique_gain": int(dnn_only),
        "dnn_unique_gain_pct": round(dnn_only / total_fraud * 100, 2) if total_fraud else 0.0,
    }


def ranking_complement(df: pd.DataFrame, cb_low_quantile: float = 0.50) -> dict:
    """
    Threshold-independent view.

    Take the frauds that CatBoost ranks in its LOWER half (likely to miss),
    and see whether the DNN ranks those same frauds highly.

    If the DNN gives high scores to frauds CatBoost ranks low,
    that is genuine complementary signal.
    """
    fraud = df[df["y_true"] == 1].copy()

    # CatBoost's low-confidence frauds (bottom half by CatBoost score)
    cb_cut = fraud["catboost_probability"].quantile(cb_low_quantile)
    cb_low_fraud = fraud[fraud["catboost_probability"] <= cb_cut]

    # Among the full fraud set, what DNN score is "high"? use DNN median on fraud.
    dnn_high_cut = fraud["dnn_probability"].median()

    rescued = (cb_low_fraud["dnn_probability"] > dnn_high_cut).sum()

    return {
        "catboost_low_fraud_count": int(len(cb_low_fraud)),
        "catboost_low_cut_value": round(float(cb_cut), 4),
        "dnn_high_cut_value": round(float(dnn_high_cut), 4),
        "dnn_rescued_from_cb_low": int(rescued),
        "dnn_rescue_pct": round(rescued / len(cb_low_fraud) * 100, 2) if len(cb_low_fraud) else 0.0,
    }


def main():
    print("\n" + "=" * 70)
    print("DNN vs CATBOOST DISAGREEMENT / COMPLEMENTARY COVERAGE")
    print("=" * 70)

    df = load_merged()
    fraud_rate = df["y_true"].mean()
    print(f"Val rows: {len(df)} | fraud: {df['y_true'].sum()} | fraud rate: {fraud_rate:.4f}")

    # CatBoost fixed at 0.50 baseline.
    # DNN tested at several thresholds because its scale differs.
    print("\n--- Fraud coverage overlap at different DNN thresholds ---")
    rows = []
    for dnn_t in [0.30, 0.40, 0.50]:
        rows.append(coverage_at_thresholds(df, cb_threshold=0.50, dnn_threshold=dnn_t))
    cov = pd.DataFrame(rows)
    print(cov.to_string(index=False))

    print("\n--- Threshold-independent ranking complement ---")
    rank = ranking_complement(df)
    for k, v in rank.items():
        print(f"  {k}: {v}")

    print("\n" + "-" * 70)
    print("HOW TO READ THIS")
    print("-" * 70)
    print("  caught_by_dnn_only  -> frauds DNN catches that CatBoost (at 0.50) misses.")
    print("  dnn_rescued_from_cb_low -> frauds CatBoost ranks low but DNN ranks high.")
    print("  If both are near zero -> DNN is redundant; drop from blend.")
    print("  If meaningfully > 0 -> DNN has complementary value for escalation.")


if __name__ == "__main__":
    main()