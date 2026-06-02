from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# -------------------------------------------------------------------
# MODEL-STAGE OUTPUT
# Used by CatBoost / DNN / Autoencoder inference modules
# -------------------------------------------------------------------
@dataclass
class ModelScores:
    catboost_probability: Optional[float] = None
    dnn_probability: Optional[float] = None
    reconstruction_error: Optional[float] = None
    anomaly_score: Optional[float] = None
    anomaly_flag: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------
# FUSION OUTPUT
# Used after CatBoost + DNN fusion, and optional anomaly combination
# -------------------------------------------------------------------
@dataclass
class FusionResult:
    supervised_score: float
    transaction_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------
# EXPLAINABILITY ITEM
# Individual SHAP-based explanation row
# -------------------------------------------------------------------
@dataclass
class ExplanationItem:
    feature_name: str
    shap_value: float
    feature_value: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------
# MAIN TRANSACTION-LEVEL RESULT
# This is the core scored output per transaction
# -------------------------------------------------------------------
@dataclass
class TransactionResult:
    transaction_id: str
    entity_id: str

    # Model outputs
    catboost_probability: Optional[float] = None
    dnn_probability: Optional[float] = None
    reconstruction_error: Optional[float] = None
    anomaly_score: Optional[float] = None
    anomaly_flag: Optional[bool] = None

    # Fusion / final scoring
    supervised_score: Optional[float] = None
    transaction_risk: Optional[float] = None

    # Decision layer
    risk_tier: Optional[str] = None
    recommended_action: Optional[str] = None
    escalation_reason: Optional[str] = None

    # Explainability
    top_shap_features: List[ExplanationItem] = field(default_factory=list)

    # Metadata
    model_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["top_shap_features"] = [item.to_dict() for item in self.top_shap_features]
        return payload


# -------------------------------------------------------------------
# ENTITY PROFILE
# Dynamic risk state for a UID/entity
# -------------------------------------------------------------------
@dataclass
class EntityProfile:
    entity_id: str
    entity_risk: float = 0.0
    entity_status: str = "Clean"

    transaction_count: int = 0
    high_risk_count: int = 0
    anomaly_count: int = 0
    review_count: int = 0
    block_count: int = 0

    risk_history: List[float] = field(default_factory=list)
    anomaly_history: List[bool] = field(default_factory=list)

    first_seen: Optional[int] = None
    last_seen: Optional[int] = None

    risk_trend_flag: bool = False
    early_warning_flag: bool = False
    last_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------
# ESCALATION RESULT
# Output after applying escalation rules
# -------------------------------------------------------------------
@dataclass
class EscalationResult:
    final_action: str
    escalated: bool
    escalation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------
# EARLY WARNING RESULT
# Output after checking warning rules over entity behaviour
# -------------------------------------------------------------------
@dataclass
class EarlyWarningResult:
    warning_triggered: bool
    warnings: List[str] = field(default_factory=list)
    recent_anomaly_count: int = 0
    consecutive_medium_count: int = 0
    risk_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------
# EXPLANATION RESULT
# CatBoost SHAP explanation for a transaction
# -------------------------------------------------------------------
@dataclass
class ExplanationResult:
    transaction_id: str
    top_features: List[ExplanationItem] = field(default_factory=list)
    reason_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["top_features"] = [item.to_dict() for item in self.top_features]
        return payload


# -------------------------------------------------------------------
# OPTIONAL PIPELINE BUNDLE
# Useful later for pipeline.py / Streamlit / API responses
# -------------------------------------------------------------------
@dataclass
class PipelineResult:
    transaction: TransactionResult
    entity_profile: Optional[EntityProfile] = None
    escalation: Optional[EscalationResult] = None
    early_warning: Optional[EarlyWarningResult] = None
    explanation: Optional[ExplanationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction": self.transaction.to_dict(),
            "entity_profile": self.entity_profile.to_dict() if self.entity_profile else None,
            "escalation": self.escalation.to_dict() if self.escalation else None,
            "early_warning": self.early_warning.to_dict() if self.early_warning else None,
            "explanation": self.explanation.to_dict() if self.explanation else None,
        }
# For config

# “Do I need a new path, threshold, weight, or runtime setting?”

# For schema

# “Am I introducing a truly new kind of output, or just filling an existing field?”

# Config grows gradually with each module by adding paths, thresholds, and tunable parameters. Schema should remain mostly stable; later modules should mainly populate existing fields instead of forcing redesign.

# Schema note: During implementation, use modular stage-based schemas for clarity and debugging. 
# After the full pipeline is stable, add a final-output-oriented wrapper for dashboard/API/export use, without removing the internal modular schemas. Make it more final product contract oriented