"""
core/fusion_tune.py

One-time weight tuning for supervised fusion.

Goal:
- find the CatBoost/DNN weight pair (w1, w2) that maximizes
  the validation AUC-PR of the supervised_score.

Why AUC-PR:
- the data is highly imbalanced (~3.5% fraud),
- AUC-PR reflects ranking quality on the positive (fraud) class,
- it is threshold-independent, so we tune ranking first and
  handle the decision threshold in a later, separate step.

Why one-time:
- weights only need re-tuning when the underlying models change.
- this script PRINTS the best weights; you update config.py manually.
  (Manual update keeps config changes explicit and reviewable.)

Data source:
- reads existing val prediction CSVs produced by model_check.py.
- no re-inference: same models + same val data = identical scores.

Note:
- the autoencoder anomaly score is NOT included here. We tune the
  supervised weights on the supervised score alone, because blending
  the weak auxiliary signal into the optimization target would
  distort the weight search. The anomaly blend stays at config values.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import utils.config as config


def load_supervised_scores() -> pd.DataFrame:
    """
    Load CatBoost and DNN val predictions and merge on transaction_id.

    Merging on transaction_id (not row order) guards against any row
    misalignment between the two CSVs.
    """
    cb = pd.read_csv(config.CATBOOST_VAL_PREDICTIONS_FILE)
    dnn = pd.read_csv(config.DNN_VAL_PREDICTIONS_FILE)

    cb = cb[["transaction_id", "catboost_probability", "y_true"]]
    dnn = dnn[["transaction_id", "dnn_probability"]]

    merged = cb.merge(dnn, on="transaction_id", how="inner")

    if len(merged) != len(cb) or len(merged) != len(dnn):
        print(
            f"[WARN] Row count changed after merge: "
            f"catboost={len(cb)}, dnn={len(dnn)}, merged={len(merged)}. "
            f"Some transaction_ids did not match."
        )

    return merged


def tune_weights(
    df: pd.DataFrame,
    w1_min: float = 0.50,
    w1_max: float = 0.90,
    step: float = 0.05,
) -> pd.DataFrame:
    """
    Grid search over CatBoost weight w1 (w2 = 1 - w1).

    For each pair, compute supervised_score and its val AUC-PR.
    Returns a results table sorted by AUC-PR descending.
    """
    y = df["y_true"].values.astype(int)
    cb = df["catboost_probability"].values.astype(float)
    dnn = df["dnn_probability"].values.astype(float)

    rows = []
    w1 = w1_min
    while w1 <= w1_max + 1e-9:
        w2 = 1.0 - w1
        supervised = (w1 * cb) + (w2 * dnn)

        auc_pr = average_precision_score(y, supervised)
        auc_roc = roc_auc_score(y, supervised)

        rows.append({
            "catboost_weight": round(w1, 4),
            "dnn_weight": round(w2, 4),
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
        })
        w1 += step

    results = pd.DataFrame(rows).sort_values("auc_pr", ascending=False).reset_index(drop=True)
    return results


def main():
    print("\n" + "=" * 70)
    print("FUSION WEIGHT TUNING (supervised: CatBoost + DNN)")
    print("=" * 70)

    df = load_supervised_scores()
    print(f"Loaded {len(df)} val rows | fraud rate: {df['y_true'].mean():.4f}")

    # Baselines for comparison
    y = df["y_true"].values.astype(int)
    cb_only = average_precision_score(y, df["catboost_probability"].values)
    dnn_only = average_precision_score(y, df["dnn_probability"].values)
    print(f"\nBaseline AUC-PR | CatBoost alone: {cb_only:.4f} | DNN alone: {dnn_only:.4f}")

    results = tune_weights(df)

    print("\nGrid search results (sorted by AUC-PR):")
    print(results.to_string(index=False))

    best = results.iloc[0]
    print("\n" + "-" * 70)
    print("BEST WEIGHTS FOUND")
    print("-" * 70)
    print(f"  CATBOOST_WEIGHT = {best['catboost_weight']}")
    print(f"  DNN_WEIGHT      = {best['dnn_weight']}")
    print(f"  Fused AUC-PR    = {best['auc_pr']:.4f}")
    print(f"  Fused AUC-ROC   = {best['auc_roc']:.4f}")

    # Honest comparison: did fusion actually beat CatBoost alone?
    delta = best["auc_pr"] - cb_only
    if delta > 0:
        print(f"\n  Fusion improves over CatBoost alone by +{delta:.4f} AUC-PR.")
    else:
        print(f"\n  Fusion does NOT beat CatBoost alone ({delta:.4f}). "
              f"Consider keeping CatBoost-dominant weights or CatBoost only.")

    print("\nNext step: manually update CATBOOST_WEIGHT and DNN_WEIGHT in config.py.")


if __name__ == "__main__":
    main()