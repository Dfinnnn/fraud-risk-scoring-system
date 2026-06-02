"""
core/disagreement_cost.py

Purpose:
- measure the FALSE POSITIVE cost of a DNN-disagreement escalation trigger,
  so we can decide whether the trigger is worth using.

Background:
- earlier analysis showed DNN catches frauds CatBoost misses
  (complementary coverage). But catching extra fraud is only useful
  if it does not flood the system with false reviews.

Trigger definition:
- "disagreement escalation" fires when:
      CatBoost score is LOW (below cb_threshold)  AND
      DNN score is HIGH (at or above dnn_threshold)
- this targets exactly the cases where CatBoost would clear a
  transaction but DNN flags it.

What we measure, for each DNN threshold:
- extra_fraud_caught : frauds the trigger newly catches (true positives gained)
- extra_false_alarms : non-frauds the trigger newly flags (false positives added)
- cost_ratio         : false alarms per extra fraud caught
                       (lower is better; high means the trigger is too noisy)

Decision guide:
- a trigger that catches N frauds at a reasonable false-alarm cost is viable.
- a trigger with a very high cost_ratio is not worth the review burden.
"""

import numpy as np
import pandas as pd

import utils.config as config


def load_merged() -> pd.DataFrame:
    cb = pd.read_csv(config.CATBOOST_VAL_PREDICTIONS_FILE)
    dnn = pd.read_csv(config.DNN_VAL_PREDICTIONS_FILE)

    cb = cb[["transaction_id", "catboost_probability", "y_true"]]
    dnn = dnn[["transaction_id", "dnn_probability"]]
    return cb.merge(dnn, on="transaction_id", how="inner")


def trigger_cost(df: pd.DataFrame, cb_threshold: float, dnn_threshold: float) -> dict:
    """
    Evaluate the disagreement trigger:
        fires when catboost < cb_threshold AND dnn >= dnn_threshold.
    """
    cb = df["catboost_probability"].values
    dnn = df["dnn_probability"].values
    y = df["y_true"].values.astype(int)

    # CatBoost would CLEAR these (below threshold) = the cases at risk of being missed
    cb_clears = cb < cb_threshold

    # Trigger fires: CatBoost clears but DNN flags
    trigger = cb_clears & (dnn >= dnn_threshold)

    # Among triggered: how many are actually fraud vs not
    extra_fraud_caught = int((trigger & (y == 1)).sum())
    extra_false_alarms = int((trigger & (y == 0)).sum())
    total_triggered = int(trigger.sum())

    cost_ratio = (extra_false_alarms / extra_fraud_caught) if extra_fraud_caught else float("inf")

    # For context: how many frauds does CatBoost alone miss at this threshold
    cb_missed_fraud = int((cb_clears & (y == 1)).sum())

    return {
        "dnn_threshold": dnn_threshold,
        "total_triggered": total_triggered,
        "extra_fraud_caught": extra_fraud_caught,
        "extra_false_alarms": extra_false_alarms,
        "cost_ratio_fp_per_fraud": round(cost_ratio, 2),
        "cb_missed_fraud_at_threshold": cb_missed_fraud,
        "pct_of_cb_misses_recovered": round(extra_fraud_caught / cb_missed_fraud * 100, 2)
        if cb_missed_fraud else 0.0,
    }

def trigger_cost_with_ae(df: pd.DataFrame, cb_threshold: float, dnn_threshold: float) -> dict:
    ae = pd.read_csv(config.AE_VAL_SCORES_FILE)
    ae = ae[["transaction_id", "anomaly_flag"]]
    df2 = df.merge(ae, on="transaction_id", how="inner")

    cb = df2["catboost_probability"].values
    dnn = df2["dnn_probability"].values
    y = df2["y_true"].values.astype(int)
    af = df2["anomaly_flag"].values.astype(bool)

    # Constrained trigger: CatBoost clears AND DNN flags AND anomaly_flag=True
    trigger = (cb < cb_threshold) & (dnn >= dnn_threshold) & af

    extra_fraud = int((trigger & (y == 1)).sum())
    extra_fp    = int((trigger & (y == 0)).sum())
    ratio = round(extra_fp / extra_fraud, 2) if extra_fraud else float("inf")

    return {
        "dnn_threshold": dnn_threshold,
        "extra_fraud_caught": extra_fraud,
        "extra_false_alarms": extra_fp,
        "cost_ratio_fp_per_fraud": ratio,
    }
    

def main():
    print("\n" + "=" * 70)
    print("DNN DISAGREEMENT TRIGGER - FALSE POSITIVE COST")
    print("=" * 70)

    df = load_merged()
    y = df["y_true"].values.astype(int)
    total_normal = int((y == 0).sum())
    total_fraud = int((y == 1).sum())
    print(f"Val rows: {len(df)} | fraud: {total_fraud} | normal: {total_normal}")

    cb_threshold = 0.50
    print(f"\nCatBoost clear threshold: {cb_threshold}")
    print("Trigger fires when CatBoost < 0.50 AND DNN >= dnn_threshold\n")

    rows = []
    for dnn_t in [0.30, 0.40, 0.50, 0.60, 0.70]:
        rows.append(trigger_cost(df, cb_threshold, dnn_t))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    print("\n" + "-" * 70)
    print("HOW TO READ THIS")
    print("-" * 70)
    print("  extra_fraud_caught  : true frauds the trigger recovers (good)")
    print("  extra_false_alarms  : normal txns wrongly flagged (cost)")
    print("  cost_ratio_fp_per_fraud : false alarms per fraud recovered")
    print("     - low ratio  -> efficient trigger, worth using")
    print("     - high ratio -> noisy trigger, review burden may not be worth it")
    print("\n  Compare against context: current CatBoost at 0.50 already produces")
    print("  many false positives. The trigger's added cost should be judged")
    print("  relative to the extra fraud it recovers, not in isolation.")
     # --- Constrained trigger: CatBoost clears AND DNN flags AND anomaly_flag=True ---
    print("\n--- Constrained trigger: CatBoost clears AND DNN flags AND anomaly_flag=True ---")
    ae = pd.read_csv(config.AE_VAL_SCORES_FILE)
    ae = ae[["transaction_id", "anomaly_flag"]]
    df2 = df.merge(ae, on="transaction_id", how="inner")

    rows2 = []
    for dnn_t in [0.30, 0.40, 0.50, 0.60, 0.70]:
        cb_arr = df2["catboost_probability"].values
        dnn_arr = df2["dnn_probability"].values
        y2 = df2["y_true"].values.astype(int)
        af = df2["anomaly_flag"].values.astype(bool)

        trigger = (cb_arr < 0.50) & (dnn_arr >= dnn_t) & af
        extra_fraud = int((trigger & (y2 == 1)).sum())
        extra_fp    = int((trigger & (y2 == 0)).sum())
        ratio = round(extra_fp / extra_fraud, 2) if extra_fraud else float("inf")

        rows2.append({
            "dnn_threshold": dnn_t,
            "extra_fraud_caught": extra_fraud,
            "extra_false_alarms": extra_fp,
            "cost_ratio_fp_per_fraud": ratio,
        })

    print(pd.DataFrame(rows2).to_string(index=False))


if __name__ == "__main__":
    main()