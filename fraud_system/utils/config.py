from pathlib import Path

# -------------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------------------------------------------------------
# DATA DIRECTORIES
# -------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "feature_engineered"
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "outputs"

PREDICTION_DIR = OUTPUT_DIR / "predictions"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
LOG_DIR = OUTPUT_DIR / "logs"

for path in [RAW_DIR, FEATURE_DIR, MODEL_DIR, OUTPUT_DIR, PREDICTION_DIR, EVALUATION_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# CORE IDENTIFIERS
# -------------------------------------------------------------------
TRANSACTION_ID_COL = "TransactionID"
ENTITY_ID_COL = "UID3"
TARGET_COL = "isFraud"
TIMESTAMP_COL = "TransactionDT"

# -------------------------------------------------------------------
# FEATURE ARTIFACTS
# -------------------------------------------------------------------
CATBOOST_FEATURE_LIST = FEATURE_DIR / "catboost_feature_list.csv"
CATBOOST_CAT_FEATURES = FEATURE_DIR / "catboost_cat_features.csv"
CATBOOST_CLASS_WEIGHTS = FEATURE_DIR / "catboost_class_weights.csv"

DNN_FEATURE_LIST = FEATURE_DIR / "dnn_feature_list.csv"
DNN_SCALER_PKL = FEATURE_DIR / "dnn_scaler.pkl"
DNN_FREQ_MAPS_PKL = FEATURE_DIR / "dnn_freq_maps.pkl"
DNN_SENTINEL_PKL = FEATURE_DIR / "dnn_sentinel_replacements.pkl"

AE_FEATURE_LIST = FEATURE_DIR / "autoencoder_feature_list.csv"
AE_SCALER_PKL = FEATURE_DIR / "autoencoder_scaler.pkl"
AE_FREQ_MAPS_PKL = FEATURE_DIR / "autoencoder_freq_maps.pkl"
AE_SENTINEL_PKL = FEATURE_DIR / "autoencoder_sentinel_replacements.pkl"

# -------------------------------------------------------------------
# TRAIN / VALIDATION DATASETS
# -------------------------------------------------------------------
CATBOOST_TRAIN_FILE = FEATURE_DIR / "train_catboost_ready.csv"
CATBOOST_VAL_FILE = FEATURE_DIR / "val_catboost_ready.csv"

# -------------------------------------------------------------------
# MODEL FILES
# -------------------------------------------------------------------
CATBOOST_MODEL_FILE = MODEL_DIR / "catboost_model.cbm"
CATBOOST_FEATURE_IMPORTANCE_FILE = MODEL_DIR / "catboost_feature_importance.csv"

# -------------------------------------------------------------------
# DNN DATA FILES
# -------------------------------------------------------------------
DNN_TRAIN_FILE = FEATURE_DIR / "train_dnn_ready.csv"
DNN_VAL_FILE = FEATURE_DIR / "val_dnn_ready.csv"

# -------------------------------------------------------------------
# DNN OUTPUT FILES
# -------------------------------------------------------------------
DNN_TRAINING_HISTORY_FILE = MODEL_DIR / "dnn_training_history.csv"

# -------------------------------------------------------------------
# AUTOENCODER DATA FILES
# -------------------------------------------------------------------
AE_TRAIN_FILE = FEATURE_DIR / "train_ae_ready.csv"
AE_VAL_FILE = FEATURE_DIR / "val_ae_ready.csv"

# -------------------------------------------------------------------
# AUTOENCODER OUTPUT FILES
# -------------------------------------------------------------------
AE_TRAIN_FILE = FEATURE_DIR / "train_autoencoder_normal.csv"
AE_VAL_FILE = FEATURE_DIR / "val_autoencoder_eval.csv"
AE_TRAINING_HISTORY_FILE = MODEL_DIR / "autoencoder_training_history.csv"

# Better to tune later in shared checks
AE_THRESHOLD_QUANTILE = 0.95

DNN_MODEL_FILE = MODEL_DIR / "dnn_model.keras"
AE_MODEL_FILE = MODEL_DIR / "autoencoder_model.keras"
AE_THRESHOLD_FILE = MODEL_DIR / "ae_threshold.txt"

CATBOOST_MODEL_VERSION = "catboost_v1"
DNN_MODEL_VERSION = "dnn_v1"
AE_MODEL_VERSION = "autoencoder_v1"
MODEL_VERSION = "v1"

# -------------------------------------------------------------------
# FUSION WEIGHTS
# -------------------------------------------------------------------
CATBOOST_WEIGHT = 1.0
DNN_WEIGHT = 0.0
SUPERVISED_WEIGHT = 0.80
ANOMALY_WEIGHT = 0.20

# -------------------------------------------------------------------
# ENTITY PROFILE
# -------------------------------------------------------------------
ENTITY_BETA = 0.85

# -------------------------------------------------------------------
# RISK TIERS
# -------------------------------------------------------------------
RISK_TIER_LOW_MAX = 0.40
RISK_TIER_MEDIUM_MAX = 0.70

RISK_ACTIONS = {
    "Low": "approve",
    "Medium": "review",
    "High": "block",
}

# -------------------------------------------------------------------
# ESCALATION / EARLY WARNING
# -------------------------------------------------------------------
ESCALATION_ENTITY_RISK_HIGH = 0.70
ESCALATION_CONSEC_MEDIUM = 3
ESCALATION_ANOMALY_WINDOW = 5
ESCALATION_ANOMALY_COUNT = 2
ESCALATION_RISE_WINDOW = 3
ESCALATION_RISE_THRESHOLD = 0.20

# -------------------------------------------------------------------
# EXPLAINABILITY
# -------------------------------------------------------------------
ENABLE_SHAP = True
SHAP_TOP_K = 5
SHAP_TARGET_MODEL = "catboost"

# -------------------------------------------------------------------
# RUNTIME
# -------------------------------------------------------------------
RANDOM_SEED = 42
DEVICE = "cpu"
DEBUG_MODE = True

# -------------------------------------------------------------------
# MODEL CHECK FILES
# -------------------------------------------------------------------
MODEL_CHECK_SUMMARY_FILE = EVALUATION_DIR / "model_check_summary.csv"
MODEL_CHECK_DETAIL_FILE = EVALUATION_DIR / "model_check_detail.json"

CATBOOST_VAL_PREDICTIONS_FILE = PREDICTION_DIR / "catboost_val_predictions.csv"
DNN_VAL_PREDICTIONS_FILE = PREDICTION_DIR / "dnn_val_predictions.csv"
AE_VAL_SCORES_FILE = PREDICTION_DIR / "autoencoder_val_scores.csv"

## For config

##“Do I need a new path, threshold, weight, or runtime setting?”

## For schema

##“Am I introducing a truly new kind of output, or just filling an existing field?”

## Config grows gradually with each module by adding paths, thresholds, and tunable parameters. Schema should remain mostly stable; later modules should mainly populate existing fields instead of forcing redesign.