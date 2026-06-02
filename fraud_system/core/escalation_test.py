"""
core/escalation_test.py

Purpose:
- verify that entity profiling and escalation actually fire when the
  SAME entity appears across multiple transactions.

Why this test is needed:
- the earlier 10-row smoke test had 10 unique entities, so each entity
  had only ONE transaction. With one transaction per entity:
    * entity_risk cannot build up (decay needs history),
    * high_risk_count / anomaly_count stay near zero,
    * escalation rules that depend on entity_risk or history never trigger,
    * early-warning patterns (rising trend, consecutive medium,
      anomaly burst) can never fire.
- so the smoke test proved the wiring runs, but did NOT prove the
  entity-aware logic works. This test does.

What it does:
- finds the entity with the MOST transactions in the val set,
- scores that entity's transactions in time order through the pipeline,
- prints how entity_risk, counters, tier, action, and escalation
  reasons evolve transaction by transaction,
- prints the final entity profile.

What to look for:
- entity_risk should change across transactions (decay working),
- counters (high_risk_count, anomaly_count) should increment,
- at least some transactions should show an escalation_reason or
  early-warning flag IF the entity's behaviour warrants it.
"""

import pandas as pd

import utils.config as config
from core.pipeline import FraudPipeline


def main():
    print("[Escalation Test] Loading val data...")
    val = pd.read_csv(config.CATBOOST_VAL_FILE)

    entity_col = config.ENTITY_ID_COL

    # Find the entity with the most transactions.
    counts = val[entity_col].value_counts()
    top_entity = counts.index[0]
    top_count = counts.iloc[0]
    print(f"[Escalation Test] Most frequent entity: {top_entity} "
          f"with {top_count} transactions.\n")

    # Take all transactions for that entity.
    subset = val[val[entity_col] == top_entity].copy()

    # Sort by time so the profile builds in correct order.
    if "TransactionDT" in subset.columns:
        subset = subset.sort_values("TransactionDT").reset_index(drop=True)

    print("[Escalation Test] Initialising pipeline...")
    pipe = FraudPipeline(enable_shap=False)
    print("[Escalation Test] Scoring this entity's transactions in sequence...\n")

    results = pipe.score_batch(subset, explain=False)

    # Print evolution transaction by transaction.
    print(f"{'txn_id':>10} | {'risk':>6} | {'tier':>6} | {'action':>8} | "
          f"{'anom':>5} | {'ent_risk':>8} | reason")
    print("-" * 90)

    profile = pipe.entity_store.get(str(top_entity))
    # Re-walk results; entity_risk at each step is not stored per result,
    # so we show final profile separately. Here we show per-txn decision.
    for r in results:
        print(
            f"{r.transaction_id:>10} | {r.transaction_risk:>6.3f} | "
            f"{r.risk_tier:>6} | {r.recommended_action:>8} | "
            f"{str(r.anomaly_flag):>5} | {'-':>8} | {r.escalation_reason}"
        )

    print("\n[Escalation Test] FINAL ENTITY PROFILE")
    print("-" * 50)
    p = pipe.entity_store.get(str(top_entity))
    print(f"  entity_id          : {p.entity_id}")
    print(f"  entity_risk        : {p.entity_risk:.4f}")
    print(f"  entity_status      : {p.entity_status}")
    print(f"  transaction_count  : {p.transaction_count}")
    print(f"  high_risk_count    : {p.high_risk_count}")
    print(f"  anomaly_count      : {p.anomaly_count}")
    print(f"  review_count       : {p.review_count}")
    print(f"  block_count        : {p.block_count}")
    print(f"  risk_trend_flag    : {p.risk_trend_flag}")
    print(f"  early_warning_flag : {p.early_warning_flag}")
    print(f"  last_action        : {p.last_action}")
    print(f"  risk_history       : {[round(x,3) for x in p.risk_history]}")
    print(f"  anomaly_history    : {p.anomaly_history}")


if __name__ == "__main__":
    main()