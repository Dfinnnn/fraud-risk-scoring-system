"""
core/escalation.py

Escalation and early-warning logic.

Runs AFTER entity_profile.update() for a transaction. It:
- decides the final risk tier (base tier, possibly UPGRADED),
- decides the recommended action,
- produces a human-readable escalation_reason,
- detects early-warning patterns,
- sets the EntityProfile fields that entity_profile.py deliberately
  leaves untouched:
      review_count, block_count, risk_trend_flag,
      early_warning_flag, last_action.

Design decisions (confirmed):
1. Escalation checks use the entity_risk AFTER the current transaction
   has been included (entity_profile already updated it).
2. Escalation can only UPGRADE severity, never downgrade.
   If base tier is already High, it stays High.
3. Early-warning detection uses pattern rules (rising trend, repeated
   anomalies, consecutive mediums). When a pattern fires it represents
   "kept signaling", so escalation may upgrade the tier in the SAME
   transaction (interpretation A). The flag itself is also recorded
   for monitoring.

All thresholds come from config (single source of truth).
"""

from typing import List

import utils.config as config
from utils.schema import EntityProfile


# -------------------------------------------------------------------
# Tier ordering helper (so we can enforce "upgrade only")
# -------------------------------------------------------------------
_TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}
_RANK_TIER = {0: "Low", 1: "Medium", 2: "High"}


def _max_tier(tier_a: str, tier_b: str) -> str:
    """Return the more severe of two tiers."""
    return _RANK_TIER[max(_TIER_RANK[tier_a], _TIER_RANK[tier_b])]


# -------------------------------------------------------------------
# Early-warning pattern detection
# -------------------------------------------------------------------
def detect_early_warning(profile: EntityProfile) -> List[str]:
    """
    Check the entity's recent history for early-warning patterns.
    Returns a list of triggered pattern names (empty if none).

    Patterns (from config):
    - rising trend  : entity risk rose more than RISE_THRESHOLD over
                      the last RISE_WINDOW transactions
    - anomaly burst : ANOMALY_COUNT or more anomalies cannot be checked
                      here directly (anomaly history not stored per txn),
                      so we approximate using the risk_history window for
                      the rise check and rely on escalation rule 2 for
                      single-transaction anomaly handling.
    - consecutive medium : CONSEC_MEDIUM medium-or-higher risks in a row
    """
    warnings: List[str] = []
    history = profile.risk_history

    # --- rising trend ---
    win = config.ESCALATION_RISE_WINDOW
    if len(history) > win:
        rise = history[-1] - history[-1 - win]
        if rise > config.ESCALATION_RISE_THRESHOLD:
            warnings.append("rising_risk_trend")

    # --- consecutive medium-or-higher ---
    n = config.ESCALATION_CONSEC_MEDIUM
    if len(history) >= n:
        recent = history[-n:]
        if all(r >= config.RISK_TIER_LOW_MAX for r in recent):
            warnings.append("consecutive_medium_risk")

    # --- anomaly burst: ANOMALY_COUNT+ anomalies in last ANOMALY_WINDOW txns ---
    a_win = config.ESCALATION_ANOMALY_WINDOW
    a_cnt = config.ESCALATION_ANOMALY_COUNT
    anomaly_hist = profile.anomaly_history
    if len(anomaly_hist) >= 1:
        recent_anoms = anomaly_hist[-a_win:]
        if sum(recent_anoms) >= a_cnt:
            warnings.append("anomaly_burst")

    return warnings


# -------------------------------------------------------------------
# Main escalation
# -------------------------------------------------------------------
def apply_escalation(
    base_tier: str,
    profile: EntityProfile,
    anomaly_flag: bool,
) -> dict:
    """
    Apply escalation rules to a single scored transaction.

    Parameters:
    - base_tier: the transaction's own risk tier from fusion ("Low"/"Medium"/"High")
    - profile: the EntityProfile AFTER entity_profile.update() ran
    - anomaly_flag: whether the AE flagged this transaction

    Returns a dict with:
        final_tier, recommended_action, escalation_reason,
        early_warning_flag, risk_trend_flag

    Side effects:
    - updates profile.review_count / block_count
    - sets profile.risk_trend_flag, profile.early_warning_flag
    - sets profile.last_action
    """
    final_tier = base_tier
    reasons: List[str] = []

    entity_high = profile.entity_risk >= config.ESCALATION_ENTITY_RISK_HIGH

    # --- Rule 1: Medium transaction + High entity risk -> High ---
    if base_tier == "Medium" and entity_high:
        final_tier = _max_tier(final_tier, "High")
        reasons.append("medium_txn_with_high_entity_risk")

    # --- Rule 2: Medium transaction + anomaly -> High ---
    if base_tier == "Medium" and anomaly_flag:
        final_tier = _max_tier(final_tier, "High")
        reasons.append("medium_txn_with_anomaly")

    # --- Rule 3: Low transaction + High entity risk -> Medium (review) ---
    if base_tier == "Low" and entity_high:
        final_tier = _max_tier(final_tier, "Medium")
        reasons.append("low_txn_with_high_entity_risk")

    # --- Early-warning patterns (interpretation A) ---
    ew_patterns = detect_early_warning(profile)
    early_warning_flag = len(ew_patterns) > 0
    risk_trend_flag = "rising_risk_trend" in ew_patterns

    # A fired pattern represents "kept signaling": allow upgrade.
    # Rising trend or consecutive medium pushes at least to Medium,
    # and if entity risk is already high, to High.
    if early_warning_flag:
        reasons.extend(ew_patterns)
        if entity_high:
            final_tier = _max_tier(final_tier, "High")
        else:
            final_tier = _max_tier(final_tier, "Medium")

    # --- Final action from final tier ---
    action = config.RISK_ACTIONS.get(final_tier, "review")

    # --- Update profile escalation fields ---
    # risk_trend_flag is a live signal: reflects only the current transaction.
    profile.risk_trend_flag = risk_trend_flag
    # early_warning_flag is sticky: once raised, it stays raised for monitoring.
    profile.early_warning_flag = profile.early_warning_flag or early_warning_flag
    profile.last_action = action

    if action == "review":
        profile.review_count += 1
    elif action == "block":
        profile.block_count += 1

    escalation_reason = "; ".join(reasons) if reasons else "none"

    return {
        "final_tier": final_tier,
        "recommended_action": action,
        "escalation_reason": escalation_reason,
        "early_warning_flag": early_warning_flag,
        "risk_trend_flag": risk_trend_flag,
    }


if __name__ == "__main__":
    # Sanity check using a manually built profile.
    from core.entity_profile import EntityProfileStore

    store = EntityProfileStore()

    # Simulate a sequence and run escalation after each update.
    sequence = [
        # (transaction_risk, base_tier, anomaly_flag, ts)
        (0.10, "Low", False, 1000),
        (0.45, "Medium", False, 1100),
        (0.50, "Medium", False, 1200),
        (0.55, "Medium", True, 1300),   # medium + anomaly -> should hit High
        (0.30, "Low", False, 1400),
    ]

    for risk, tier, anomaly, ts in sequence:
        p = store.update("UID_demo", risk, tier, anomaly, ts)
        result = apply_escalation(base_tier=tier, profile=p, anomaly_flag=anomaly)
        print(
            f"txn={p.transaction_count} | base={tier} -> final={result['final_tier']} | "
            f"action={result['recommended_action']} | "
            f"entity_risk={p.entity_risk:.3f} | "
            f"EW={result['early_warning_flag']} | "
            f"reason={result['escalation_reason']}"
        )

    print("\nFinal profile counters:")
    fp = store.get("UID_demo")
    print(f"  review_count={fp.review_count} | block_count={fp.block_count}")
    print(f"  risk_trend_flag={fp.risk_trend_flag} | early_warning_flag={fp.early_warning_flag}")
    print(f"  last_action={fp.last_action}")