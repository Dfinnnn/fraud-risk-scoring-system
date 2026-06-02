"""
core/entity_profile.py

Dynamic Entity Risk Profile engine.

Purpose:
- maintain a running risk profile for each entity (UID),
- update that profile after every scored transaction,
- provide the standing entity state that escalation and early-warning
  logic will later consume.

Design boundaries (important):
- This module owns the RISK UPDATE and BASIC COUNTERS only:
    entity_risk, transaction_count, high_risk_count, anomaly_count,
    risk_history, first_seen, last_seen, entity_status.
- It does NOT set escalation / early-warning fields
    (risk_trend_flag, early_warning_flag, review_count, block_count,
     last_action). Those belong to escalation.py, which runs after this.
  Keeping the boundary prevents two modules writing the same fields.

Storage:
- in-memory dictionary { entity_id -> EntityProfile }.
- enough for notebook validation and Streamlit session use.
- persistence can be added later without changing this logic.

Risk update formula (from system plan):
    entity_risk_t = beta * entity_risk_(t-1) + (1 - beta) * transaction_risk_t
- beta (config.ENTITY_BETA, default 0.85) gives memory of past behaviour
  while letting old risk decay.
"""

from typing import Dict, Optional

import utils.config as config
from utils.schema import EntityProfile


class EntityProfileStore:
    """
    In-memory store and updater for entity risk profiles.
    """

    def __init__(self, beta: float = config.ENTITY_BETA):
        self.beta = beta
        self._profiles: Dict[str, EntityProfile] = {}

    # ---------------------------------------------------------------
    # Access helpers
    # ---------------------------------------------------------------
    def get(self, entity_id: str) -> Optional[EntityProfile]:
        """Return the profile for an entity, or None if not seen yet."""
        return self._profiles.get(entity_id)

    def get_or_create(self, entity_id: str) -> EntityProfile:
        """Return the existing profile, or create a fresh one."""
        profile = self._profiles.get(entity_id)
        if profile is None:
            profile = EntityProfile(entity_id=entity_id)
            self._profiles[entity_id] = profile
        return profile

    def all_profiles(self) -> Dict[str, EntityProfile]:
        return self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    # ---------------------------------------------------------------
    # Core update
    # ---------------------------------------------------------------
    def update(
        self,
        entity_id: str,
        transaction_risk: float,
        risk_tier: str,
        anomaly_flag: bool = False,
        timestamp: Optional[int] = None,
    ) -> EntityProfile:
        """
        Update an entity's profile with one scored transaction.

        Parameters:
        - entity_id: the UID this transaction belongs to
        - transaction_risk: final blended risk in [0, 1] (drives entity_risk)
        - risk_tier: "Low" / "Medium" / "High" (used for counters)
        - anomaly_flag: whether the AE flagged this transaction
        - timestamp: optional TransactionDT for first/last seen tracking

        Returns the updated EntityProfile.
        """
        profile = self.get_or_create(entity_id)

        # --- entity_risk update (exponential decay) ---
        if profile.transaction_count == 0:
            # first transaction: no prior risk to blend, seed directly
            profile.entity_risk = float(transaction_risk)
        else:
            profile.entity_risk = (
                self.beta * profile.entity_risk
                + (1.0 - self.beta) * float(transaction_risk)
            )

        # --- counters ---
        profile.transaction_count += 1
        if risk_tier == "High":
            profile.high_risk_count += 1
        if anomaly_flag:
            profile.anomaly_count += 1

        # --- risk history (kept for trend / early-warning logic later) ---
        profile.risk_history.append(float(transaction_risk))
        profile.anomaly_history.append(bool(anomaly_flag))

        # --- timestamps ---
        if timestamp is not None:
            if profile.first_seen is None:
                profile.first_seen = int(timestamp)
            profile.last_seen = int(timestamp)

        # --- standing status from current entity_risk ---
        profile.entity_status = self._status_from_risk(profile.entity_risk)

        return profile

    # ---------------------------------------------------------------
    # Status mapping
    # ---------------------------------------------------------------
    @staticmethod
    def _status_from_risk(entity_risk: float) -> str:
        """
        Map standing entity_risk to a coarse status label.
        Uses the same tier thresholds as transaction risk for consistency.
        """
        if entity_risk < config.RISK_TIER_LOW_MAX:
            return "Clean"
        if entity_risk < config.RISK_TIER_MEDIUM_MAX:
            return "Watch"
        return "Risky"


if __name__ == "__main__":
    # Quick sanity check: feed a sequence of transactions for one entity.
    store = EntityProfileStore()

    sequence = [
        # (transaction_risk, risk_tier, anomaly_flag, timestamp)
        (0.10, "Low", False, 1000),
        (0.20, "Low", False, 1100),
        (0.55, "Medium", False, 1200),
        (0.80, "High", True, 1300),
        (0.85, "High", True, 1400),
    ]

    for risk, tier, anomaly, ts in sequence:
        p = store.update("UID_demo", risk, tier, anomaly, ts)
        print(
            f"txn={p.transaction_count} | "
            f"entity_risk={p.entity_risk:.4f} | "
            f"status={p.entity_status} | "
            f"high={p.high_risk_count} | anomaly={p.anomaly_count}"
        )

    print("\nfirst_seen:", store.get("UID_demo").first_seen)
    print("last_seen:", store.get("UID_demo").last_seen)
    print("risk_history:", store.get("UID_demo").risk_history)