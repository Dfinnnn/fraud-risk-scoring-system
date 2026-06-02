"""
core/fusion.py

Fusion layer for the hybrid fraud risk scoring system.

Responsibilities:
- combine CatBoost + DNN probabilities into a supervised score
- blend the auxiliary anomaly score into a final transaction risk
- assign a risk tier and recommended action

Design notes:
- Weights and blend factors come from config (single source of truth).
- CatBoost is the dominant supervised signal; DNN is a complement.
- The autoencoder is auxiliary: its anomaly_score is blended only lightly,
  and is clipped to [0, 1] so it cannot distort the risk scale.
- This module does pure scoring math. It does not load models or data.
  Model inference happens upstream; fusion only consumes scores.
"""

from typing import Optional

import utils.config as config
from utils.schema import FusionResult


def _clip01(value: float) -> float:
    """Clip a value into the [0, 1] range."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def compute_supervised_score(
    catboost_probability: float,
    dnn_probability: float = None,  # kept for signature compatibility / reporting
    catboost_weight: float = config.CATBOOST_WEIGHT,
    dnn_weight: float = config.DNN_WEIGHT,
) -> float:
    """
    Stage 1: supervised score.

    Design decision (evidence-based):
    - Empirical tuning showed that blending DNN into the supervised score
      REDUCED val AUC-PR at every weight (best blend 0.5606 vs CatBoost
      alone 0.5733). Disagreement-based triggers were also too noisy
      (10-20 false positives per extra fraud) or never fired.
    - Therefore the supervised score uses CatBoost only.
    - DNN is still computed and reported as a parallel complementary
      signal (it catches a distinct slice of frauds), but it does not
      feed the final risk score.
    """
    return float(catboost_probability)


def compute_transaction_risk(
    supervised_score: float,
    anomaly_score: Optional[float],
    supervised_weight: float = config.SUPERVISED_WEIGHT,
    anomaly_weight: float = config.ANOMALY_WEIGHT,
) -> float:
    """
    Stage 2: blend supervised score with the auxiliary anomaly score.

    transaction_risk = alpha * supervised_score + (1 - alpha) * anomaly_score

    Reason:
    - supervised fraud probability stays the main signal (alpha high),
    - anomaly adds sensitivity without overpowering the score,
    - anomaly_score is clipped to [0, 1] first, because the raw AE
      anomaly_score (error / threshold) can exceed 1.0 and would
      otherwise push transaction_risk above the valid range.

    If anomaly_score is missing, fall back to the supervised score alone.
    """
    if anomaly_score is None:
        return _clip01(supervised_score)

    anomaly_clipped = _clip01(anomaly_score)
    risk = (supervised_weight * supervised_score) + (anomaly_weight * anomaly_clipped)
    return _clip01(risk)


def assign_risk_tier(transaction_risk: float) -> str:
    """
    Map a continuous risk score to a tier using config thresholds.

    Low    : risk < RISK_TIER_LOW_MAX
    Medium : RISK_TIER_LOW_MAX <= risk < RISK_TIER_MEDIUM_MAX
    High   : risk >= RISK_TIER_MEDIUM_MAX
    """
    if transaction_risk < config.RISK_TIER_LOW_MAX:
        return "Low"
    if transaction_risk < config.RISK_TIER_MEDIUM_MAX:
        return "Medium"
    return "High"


def recommend_action(risk_tier: str) -> str:
    """Look up the base action for a tier from config."""
    return config.RISK_ACTIONS.get(risk_tier, "review")


def fuse(
    catboost_probability: float,
    dnn_probability: Optional[float] = None,  # accepted, reported upstream, not blended
    anomaly_score: Optional[float] = None,
) -> FusionResult:
    # supervised score is CatBoost only (see compute_supervised_score for reason)
    supervised = compute_supervised_score(catboost_probability)
    risk = compute_transaction_risk(supervised, anomaly_score)

    return FusionResult(
        supervised_score=float(supervised),
        transaction_risk=float(risk),
    )

if __name__ == "__main__":
    # Quick sanity check with hand values.
    demo = fuse(catboost_probability=0.80, dnn_probability=0.40, anomaly_score=0.30)
    print("supervised_score:", round(demo.supervised_score, 4))
    print("transaction_risk:", round(demo.transaction_risk, 4))
    tier = assign_risk_tier(demo.transaction_risk)
    print("risk_tier:", tier)
    print("action:", recommend_action(tier))