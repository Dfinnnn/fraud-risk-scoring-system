from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

import utils.config as config
from models.catboost_inference import CatBoostInferenceModel
from models.dnn_inference import DNNInferenceModel
from models.autoencoder_inference import AutoencoderInferenceModel


# -------------------------------------------------------------------
# Output paths
# These use config paths if added, otherwise fallback paths are used.
# -------------------------------------------------------------------

MODEL_CHECK_SUMMARY_FILE = getattr(
    config,
    "MODEL_CHECK_SUMMARY_FILE",
    config.EVALUATION_DIR / "model_check_summary.csv",
)

MODEL_CHECK_DETAIL_FILE = getattr(
    config,
    "MODEL_CHECK_DETAIL_FILE",
    config.EVALUATION_DIR / "model_check_detail.json",
)

CATBOOST_VAL_PREDICTIONS_FILE = getattr(
    config,
    "CATBOOST_VAL_PREDICTIONS_FILE",
    config.PREDICTION_DIR / "catboost_val_predictions.csv",
)

DNN_VAL_PREDICTIONS_FILE = getattr(
    config,
    "DNN_VAL_PREDICTIONS_FILE",
    config.PREDICTION_DIR / "dnn_val_predictions.csv",
)

AE_VAL_SCORES_FILE = getattr(
    config,
    "AE_VAL_SCORES_FILE",
    config.PREDICTION_DIR / "autoencoder_val_scores.csv",
)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def make_json_safe(obj):
    """
    Convert numpy/pandas objects into JSON-safe Python objects.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, pd.Series):
        return obj.to_dict()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    return obj


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, indent=4)


def save_classification_report(report_dict: Dict[str, Any], path: Path) -> None:
    """
    Save sklearn classification report dictionary as CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(path, index=True)


def print_clean_metrics(metrics: Dict[str, Any]) -> None:
    """
    Print only thesis-readable metric values.
    Avoid printing long nested classification report directly.
    """
    printable_keys = [
        "auc_roc",
        "auc_pr",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
        "reconstruction_threshold",
        "mean_normal_error",
        "mean_fraud_error",
        "fraud_normal_error_ratio",
        "anomaly_flag_rate",
    ]

    clean = {k: metrics[k] for k in printable_keys if k in metrics}
    print(clean)

    if "confusion_matrix" in metrics:
        print("confusion_matrix:", metrics["confusion_matrix"])

    if all(k in metrics for k in ["tn", "fp", "fn", "tp"]):
        print(
            f"TN={metrics['tn']} | FP={metrics['fp']} | "
            f"FN={metrics['fn']} | TP={metrics['tp']}"
        )


def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute binary classification metrics from probability scores.

    Used for:
    - CatBoost fraud probability
    - DNN fraud probability

    For fraud detection, accuracy is included but should not be treated
    as the main metric due to class imbalance.
    """
    preds = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y_true,
        preds,
        labels=[0, 1],
        target_names=["Normal", "Fraud"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "auc_pr": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": report,
    }


def load_history(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def review_history(
    history_df: pd.DataFrame,
    train_metric: str,
    val_metric: str,
) -> Dict[str, Any]:
    """
    Basic overfitting view:
    - best validation metric
    - last validation metric
    - train/validation gap at best epoch
    """
    if history_df is None or history_df.empty:
        return {}

    if val_metric not in history_df.columns:
        return {}

    best_idx = (
        history_df[val_metric].idxmin()
        if "loss" in val_metric.lower()
        else history_df[val_metric].idxmax()
    )

    out = {
        "epochs_ran": int(len(history_df)),
        "best_epoch_index": int(best_idx),
        f"best_{val_metric}": float(history_df.loc[best_idx, val_metric]),
        f"last_{val_metric}": float(history_df[val_metric].iloc[-1]),
    }

    if train_metric in history_df.columns and val_metric in history_df.columns:
        out[f"{train_metric}_at_best_epoch"] = float(history_df.loc[best_idx, train_metric])
        out[f"{val_metric}_at_best_epoch"] = float(history_df.loc[best_idx, val_metric])

        if "loss" in train_metric.lower():
            out["overfit_gap"] = float(
                history_df.loc[best_idx, val_metric]
                - history_df.loc[best_idx, train_metric]
            )
        else:
            out["overfit_gap"] = float(
                history_df.loc[best_idx, train_metric]
                - history_df.loc[best_idx, val_metric]
            )

    return out


# -------------------------------------------------------------------
# CatBoost evaluation
# -------------------------------------------------------------------

def evaluate_catboost() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("CATBOOST CHECK")
    print("=" * 70)

    val_df = pd.read_csv(config.CATBOOST_VAL_FILE)
    y_val = val_df[config.TARGET_COL].values.astype(int)

    model = CatBoostInferenceModel()
    model.load()

    pred_df = model.predict_scores(val_df)
    scores = pred_df["catboost_probability"].values.astype(float)

    threshold = 0.5
    metrics = compute_binary_metrics(y_val, scores, threshold=threshold)

    pred_df["y_true"] = y_val
    pred_df["pred_label"] = (scores >= threshold).astype(int)

    CATBOOST_VAL_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(CATBOOST_VAL_PREDICTIONS_FILE, index=False)

    report_path = config.EVALUATION_DIR / "catboost_classification_report.csv"
    save_classification_report(metrics["classification_report"], report_path)

    print_clean_metrics(metrics)

    print("\nSample predictions:")
    print(pred_df.head(5).to_string(index=False))

    print("\nSaved:")
    print(" - Prediction file:", CATBOOST_VAL_PREDICTIONS_FILE)
    print(" - Classification report:", report_path)

    return {
        "model": "CatBoost",
        "metrics": metrics,
        "sample_predictions": pred_df.head(5),
        "prediction_file": str(CATBOOST_VAL_PREDICTIONS_FILE),
        "classification_report_file": str(report_path),
    }

# -------------------------------------------------------------------
# DEBUG: Pre-flight data and scaler alignment check
# -------------------------------------------------------------------

def debug_preflight_check() -> None:
    print("\n" + "=" * 70)
    print("DEBUG PRE-FLIGHT CHECK")
    print("=" * 70)

    import pickle

    # --- CHECK 1: DNN scaler vs CSV column order ---
    print("\n[CHECK 1] DNN scaler column alignment")
    dnn_val = pd.read_csv(config.DNN_VAL_FILE)
    dnn_feature_list = pd.read_csv(config.DNN_FEATURE_LIST)["feature"].tolist()
    dnn_scaler = pickle.load(open(config.DNN_SCALER_PKL, "rb"))

    print(f"  Scaler expects col[0]: {dnn_feature_list[0]}")
    print(f"  CSV actual col[0]:     {dnn_val.columns[0]}")
    print(f"  Order match: {dnn_val[dnn_feature_list].columns.tolist() == dnn_feature_list}")
    print(f"  Scaler mean[0]: {dnn_scaler.mean_[0]:.4f}  (should match mean of {dnn_feature_list[0]})")
    print(f"  CSV {dnn_feature_list[0]} mean: {dnn_val[dnn_feature_list[0]].mean():.4f}")

    # --- CHECK 2: DNN val data scale range ---
    print("\n[CHECK 2] DNN val data scale range (expect roughly -4 to +4 if scaled)")
    dnn_val_features = dnn_val[dnn_feature_list]
    print(f"  Global min: {dnn_val_features.min().min():.4f}")
    print(f"  Global max: {dnn_val_features.max().max():.4f}")
    print(f"  TransactionAmt mean: {dnn_val['TransactionAmt'].mean():.4f}")

    # --- CHECK 3: DNN train data scale range ---
    print("\n[CHECK 3] DNN train data scale range (expect roughly -4 to +4 if scaled)")
    dnn_train = pd.read_csv(config.DNN_TRAIN_FILE)
    dnn_train_features = dnn_train[dnn_feature_list]
    print(f"  Global min: {dnn_train_features.min().min():.4f}")
    print(f"  Global max: {dnn_train_features.max().max():.4f}")
    print(f"  Class distribution:\n{dnn_train[config.TARGET_COL].value_counts()}")

    # --- CHECK 4: DNN training history convergence ---
    print("\n[CHECK 4] DNN training history")
    history_path = config.MODEL_DIR / "dnn_training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        print(history[["loss", "val_loss"]].to_string())
        print(f"  Final train loss:  {history['loss'].iloc[-1]:.6f}")
        print(f"  Final val loss:    {history['val_loss'].iloc[-1]:.6f}")
        if "auc" in history.columns:
            print(f"  Final train AUC:   {history['auc'].iloc[-1]:.4f}")
            print(f"  Final val AUC:     {history['val_auc'].iloc[-1]:.4f}")
    else:
        print("  WARNING: dnn_training_history.csv not found")

    # --- CHECK 5: Autoencoder scaler column alignment ---
    print("\n[CHECK 5] Autoencoder scaler column alignment")
    ae_val = pd.read_csv(config.AE_VAL_FILE)
    ae_feature_list = pd.read_csv(config.AE_FEATURE_LIST)["feature"].tolist()
    ae_scaler = pickle.load(open(config.AE_SCALER_PKL, "rb"))

    print(f"  Scaler expects col[0]: {ae_feature_list[0]}")
    print(f"  CSV actual col[0]:     {ae_val.columns[0]}")
    print(f"  Order match: {ae_val[ae_feature_list].columns.tolist() == ae_feature_list}")
    print(f"  Scaler mean[0]: {ae_scaler.mean_[0]:.4f}  (should match mean of {ae_feature_list[0]})")

    # --- CHECK 6: Autoencoder fraud vs normal reconstruction error direction ---
    print("\n[CHECK 6] Autoencoder reconstruction error direction")
    ae_model = AutoencoderInferenceModel()
    ae_model.load()
    score_df = ae_model.score_df(ae_val.head(2000))  # sample for speed
    y_sample = ae_val[config.TARGET_COL].values[:2000]

    normal_errors = score_df.loc[y_sample == 0, "reconstruction_error"].values
    fraud_errors  = score_df.loc[y_sample == 1, "reconstruction_error"].values

    print(f"  Mean normal error: {normal_errors.mean():.6f}")
    print(f"  Mean fraud error:  {fraud_errors.mean():.6f}")
    print(f"  Fraud > Normal (expected True): {fraud_errors.mean() > normal_errors.mean()}")
    print(f"  Fraud/Normal ratio: {fraud_errors.mean() / normal_errors.mean():.4f}  (expect > 1.2)")

    # --- CHECK 7: Autoencoder BatchNorm running statistics ---
    print("\n[CHECK 7] Autoencoder BatchNorm running statistics")
    import tensorflow as tf
    keras_model = tf.keras.models.load_model(str(config.AE_MODEL_FILE))
    for layer in keras_model.layers:
        if "batch_norm" in layer.name.lower() or "batch_normalization" in layer.name.lower():
            moving_mean = layer.moving_mean.numpy()
            moving_var  = layer.moving_variance.numpy()
            print(f"  Layer: {layer.name}")
            print(f"    moving_mean[:3]: {moving_mean[:3]}")
            print(f"    moving_var[:3]:  {moving_var[:3]}")
            print(f"    Mean far from 0: {(abs(moving_mean) > 1.0).sum()} / {len(moving_mean)} units")

    print("\n" + "=" * 70)
    print("PRE-FLIGHT CHECK COMPLETE")
    print("=" * 70)

# -------------------------------------------------------------------
# DNN evaluation
# -------------------------------------------------------------------

def evaluate_dnn() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("DNN CHECK")
    print("=" * 70)

    val_df = pd.read_csv(config.FEATURE_DIR / "val_step10_pruned.csv")
    y_val = val_df[config.TARGET_COL].values.astype(int)

    model = DNNInferenceModel()
    model.load()

    pred_df = model.predict_scores(val_df)
    scores = pred_df["dnn_probability"].values.astype(float)

    threshold = 0.5
    metrics = compute_binary_metrics(y_val, scores, threshold=threshold)

    pred_df["y_true"] = y_val
    pred_df["pred_label"] = (scores >= threshold).astype(int)

    DNN_VAL_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(DNN_VAL_PREDICTIONS_FILE, index=False)

    report_path = config.EVALUATION_DIR / "dnn_classification_report.csv"
    save_classification_report(metrics["classification_report"], report_path)

    history_path = getattr(
        config,
        "DNN_TRAINING_HISTORY_FILE",
        config.MODEL_DIR / "dnn_training_history.csv",
    )

    history_df = load_history(history_path)
    history_review = (
        review_history(history_df, train_metric="auc_pr", val_metric="val_auc_pr")
        if history_df is not None
        else {}
    )

    print_clean_metrics(metrics)

    if history_review:
        print("\nHistory review:")
        print(history_review)

    print("\nSample predictions:")
    print(pred_df.head(5).to_string(index=False))

    print("\nSaved:")
    print(" - Prediction file:", DNN_VAL_PREDICTIONS_FILE)
    print(" - Classification report:", report_path)

    return {
        "model": "DNN",
        "metrics": metrics,
        "history_review": history_review,
        "sample_predictions": pred_df.head(5),
        "prediction_file": str(DNN_VAL_PREDICTIONS_FILE),
        "classification_report_file": str(report_path),
    }


# -------------------------------------------------------------------
# Autoencoder evaluation
# -------------------------------------------------------------------

def evaluate_autoencoder() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("AUTOENCODER CHECK")
    print("=" * 70)

    val_df = pd.read_csv(config.FEATURE_DIR / "val_step10_pruned.csv")
    y_val = val_df[config.TARGET_COL].values.astype(int)

    model = AutoencoderInferenceModel()
    model.load()

    score_df = model.score_df(val_df)

    anomaly_scores = score_df["anomaly_score"].values.astype(float)
    anomaly_flags = score_df["anomaly_flag"].astype(int).values

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        anomaly_flags,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_val, anomaly_flags, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y_val,
        anomaly_flags,
        labels=[0, 1],
        target_names=["Normal", "Fraud"],
        zero_division=0,
        output_dict=True,
    )

    fraud_mask = y_val == 1
    normal_mask = y_val == 0

    fraud_errors = score_df.loc[fraud_mask, "reconstruction_error"].values
    normal_errors = score_df.loc[normal_mask, "reconstruction_error"].values

    reconstruction_threshold = (
        float(model._threshold)
        if hasattr(model, "_threshold") and model._threshold is not None
        else None
    )

    metrics = {
        "auc_roc": float(roc_auc_score(y_val, anomaly_scores)),
        "auc_pr": float(average_precision_score(y_val, anomaly_scores)),
        "accuracy": float(accuracy_score(y_val, anomaly_flags)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": 1.0,
        "reconstruction_threshold": reconstruction_threshold,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": report,
        "mean_normal_error": float(np.mean(normal_errors)) if len(normal_errors) else None,
        "mean_fraud_error": float(np.mean(fraud_errors)) if len(fraud_errors) else None,
        "fraud_normal_error_ratio": (
            float(np.mean(fraud_errors) / np.mean(normal_errors))
            if len(fraud_errors) and len(normal_errors) and np.mean(normal_errors) > 0
            else None
        ),
        "anomaly_flag_rate": float(np.mean(anomaly_flags)),
    }

    score_df["y_true"] = y_val

    AE_VAL_SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(AE_VAL_SCORES_FILE, index=False)

    report_path = config.EVALUATION_DIR / "autoencoder_classification_report.csv"
    save_classification_report(metrics["classification_report"], report_path)

    history_path = getattr(
        config,
        "AE_TRAINING_HISTORY_FILE",
        config.MODEL_DIR / "autoencoder_training_history.csv",
    )

    history_df = load_history(history_path)
    history_review = (
        review_history(history_df, train_metric="loss", val_metric="val_loss")
        if history_df is not None
        else {}
    )

    print_clean_metrics(metrics)

    if history_review:
        print("\nHistory review:")
        print(history_review)

    print("\nSample anomaly scores:")
    print(score_df.head(5).to_string(index=False))

    print("\nSaved:")
    print(" - Autoencoder score file:", AE_VAL_SCORES_FILE)
    print(" - Classification report:", report_path)

    return {
        "model": "Autoencoder",
        "metrics": metrics,
        "history_review": history_review,
        "sample_scores": score_df.head(5),
        "score_file": str(AE_VAL_SCORES_FILE),
        "classification_report_file": str(report_path),
    }


# -------------------------------------------------------------------
# Summary output
# -------------------------------------------------------------------

def build_summary_table(
    catboost_result: Dict[str, Any],
    dnn_result: Dict[str, Any],
    ae_result: Dict[str, Any],
) -> pd.DataFrame:
    """
    Create thesis-ready summary table.
    Autoencoder is included, but it should be explained as an anomaly signal.
    """
    rows = []

    for result in [catboost_result, dnn_result, ae_result]:
        metrics = result["metrics"]

        rows.append({
            "model": result["model"],
            "auc_roc": metrics.get("auc_roc"),
            "auc_pr": metrics.get("auc_pr"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "threshold": metrics.get("threshold"),
            "tn": metrics.get("tn"),
            "fp": metrics.get("fp"),
            "fn": metrics.get("fn"),
            "tp": metrics.get("tp"),
            "mean_normal_error": metrics.get("mean_normal_error"),
            "mean_fraud_error": metrics.get("mean_fraud_error"),
            "fraud_normal_error_ratio": metrics.get("fraud_normal_error_ratio"),
            "anomaly_flag_rate": metrics.get("anomaly_flag_rate"),
        })

    summary_df = pd.DataFrame(rows)
    return summary_df


# -------------------------------------------------------------------
# Tuning recommendations
# -------------------------------------------------------------------

def make_tuning_recommendations(
    catboost_result: Dict[str, Any],
    dnn_result: Dict[str, Any],
    ae_result: Dict[str, Any],
) -> None:
    print("\n" + "=" * 70)
    print("TUNING RECOMMENDATIONS")
    print("=" * 70)

    cb = catboost_result["metrics"]
    dnn = dnn_result["metrics"]
    dnn_hist = dnn_result.get("history_review", {})
    ae = ae_result["metrics"]

    print("\n[CatBoost]")
    if cb["recall"] < 0.60:
        print("- Recall is still weak. First adjust decision threshold before changing the model.")
    else:
        print("- CatBoost looks stable enough as the main structured model. Keep as baseline for fusion.")

    print("\n[DNN]")
    if dnn_hist:
        gap = dnn_hist.get("overfit_gap")

        if gap is not None and gap > 0.05:
            print("- DNN shows overfitting risk. First try smaller layers or stronger dropout.")
        else:
            print("- DNN train/validation gap looks acceptable.")

    if dnn["auc_pr"] < cb["auc_pr"]:
        print("- DNN is weaker than CatBoost. Keep it as behavioural complement, not dominant model.")
    else:
        print("- DNN is adding meaningful signal and is worth keeping for fusion.")

    print("\n[Autoencoder]")
    ratio = ae.get("fraud_normal_error_ratio")

    if ratio is not None and ratio < 1.2:
        print("- Fraud/normal reconstruction separation is weak. Tune AE before trusting it in escalation.")
        print("- First change threshold quantile and inspect error distribution before changing architecture.")
        print("- If still weak, remove BatchNorm or reduce bottleneck size.")
    else:
        print("- AE separation looks useful enough for auxiliary anomaly signalling.")

    if ae["recall"] < 0.10:
        print("- AE recall is very low. Use it only as a weak auxiliary signal until tuned.")
    elif ae["precision"] < 0.05:
        print("- AE may be too noisy. Increase threshold quantile or reduce anomaly sensitivity.")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    catboost_result = evaluate_catboost()
    dnn_result = evaluate_dnn()
    ae_result = evaluate_autoencoder()

    summary_df = build_summary_table(catboost_result, dnn_result, ae_result)

    MODEL_CHECK_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(MODEL_CHECK_SUMMARY_FILE, index=False)

    detail_payload = {
        "catboost": catboost_result,
        "dnn": dnn_result,
        "autoencoder": ae_result,
        "notes": {
            "accuracy_note": (
                "Accuracy is reported as a supporting metric only because fraud detection "
                "is highly imbalanced. AUC-PR, recall, precision, and F1-score should be "
                "prioritised in Chapter 4 discussion."
            ),
            "autoencoder_note": (
                "The Autoencoder is evaluated as an anomaly detection signal using "
                "reconstruction error and anomaly score, not as an equal supervised "
                "fraud probability classifier."
            ),
            "current_scope_note": (
                "These results represent preliminary individual model checking only. "
                "Fusion, final transaction risk score, risk tiers, entity profiling, "
                "escalation logic, early warning, and SHAP explanations are not evaluated here."
            ),
        },
    }

    save_json(detail_payload, MODEL_CHECK_DETAIL_FILE)

    print("\n" + "=" * 70)
    print("MODEL CHECK SUMMARY TABLE")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\nSaved summary outputs:")
    print(" - Summary CSV:", MODEL_CHECK_SUMMARY_FILE)
    print(" - Detail JSON:", MODEL_CHECK_DETAIL_FILE)

    make_tuning_recommendations(catboost_result, dnn_result, ae_result)


if __name__ == "__main__":
    debug_preflight_check() 
    main()
