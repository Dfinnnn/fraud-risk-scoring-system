# plot_ch4_outputs.py
# ------------------------------------------------------------
# Chapter 4.3 Output Extraction and Visualization
#
# Purpose:
# - Extract thesis-ready tables
# - Plot CatBoost feature importance
# - Plot DNN training curves
# - Plot Autoencoder training curves
# - Plot Precision-Recall and ROC comparison curves
# - Print short sample outputs for Chapter 4.4
#
# Note:
# - This script does NOT save figures by default.
# - It only displays plots using plt.show().
# - Run model_check.py first before running this script.
# ------------------------------------------------------------

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)

import utils.config as config


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

TOP_N_FEATURES = 15
SHOW_SAMPLE_ROWS = 5

# No saving by default, as requested
SAVE_FIGURES = False

# Only used if SAVE_FIGURES = True
FIGURE_DIR = config.OUTPUT_DIR / "chapter4_figures"


# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

SUMMARY_FILE = config.EVALUATION_DIR / "model_check_summary.csv"

CATBOOST_IMPORTANCE_FILE = config.MODEL_DIR / "catboost_feature_importance.csv"
DNN_HISTORY_FILE = config.MODEL_DIR / "dnn_training_history.csv"
AE_HISTORY_FILE = config.MODEL_DIR / "autoencoder_training_history.csv"
AE_THRESHOLD_FILE = config.MODEL_DIR / "ae_threshold.txt"

CATBOOST_PRED_FILE = config.PREDICTION_DIR / "catboost_val_predictions.csv"
DNN_PRED_FILE = config.PREDICTION_DIR / "dnn_val_predictions.csv"
AE_SCORE_FILE = config.PREDICTION_DIR / "autoencoder_val_scores.csv"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def read_csv_required(path: Path, file_description: str) -> pd.DataFrame:
    """
    Load a required CSV file and provide a clear error message if missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{file_description} not found:\n{path}\n\n"
            f"Run the relevant training/model_check script first."
        )

    return pd.read_csv(path)


def read_text_optional(path: Path):
    """
    Load optional text file such as ae_threshold.txt.
    """
    if not path.exists():
        return None

    return path.read_text().strip()


def save_or_show(fig_name: str):
    """
    Show figure by default. Save only if SAVE_FIGURES=True.
    """
    plt.tight_layout()

    if SAVE_FIGURES:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURE_DIR / f"{fig_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {output_path}")

    plt.show()


def check_probability_range(df: pd.DataFrame, score_col: str, model_name: str):
    """
    Check whether supervised model probability scores are between 0 and 1.
    This is important because DNN and CatBoost should output probabilities.
    """
    if score_col not in df.columns:
        raise ValueError(f"{score_col} column not found in {model_name} prediction file.")

    scores = pd.to_numeric(df[score_col], errors="coerce")

    if scores.isna().any():
        warnings.warn(
            f"{model_name}: Some values in {score_col} are not numeric. "
            f"Please check the prediction CSV."
        )

    min_score = scores.min()
    max_score = scores.max()

    if min_score < 0 or max_score > 1:
        warnings.warn(
            f"{model_name}: {score_col} contains values outside [0, 1]. "
            f"Min={min_score}, Max={max_score}. "
            f"Please check whether the CSV columns are shifted or incorrectly exported."
        )


def require_columns(df: pd.DataFrame, required_cols, file_name: str):
    """
    Validate required columns.
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {file_name}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


# ------------------------------------------------------------
# Table 4.x: Individual Model Validation Performance
# ------------------------------------------------------------

def show_model_performance_table():
    summary_df = read_csv_required(
        SUMMARY_FILE,
        "Model check summary file"
    )

    selected_cols = [
        "model",
        "auc_roc",
        "auc_pr",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    available_cols = [col for col in selected_cols if col in summary_df.columns]
    table_df = summary_df[available_cols].copy()

    metric_cols = [
        "auc_roc",
        "auc_pr",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
    ]

    for col in metric_cols:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").round(4)

    print("\n" + "=" * 80)
    print("Table 4.x Individual Model Validation Performance")
    print("=" * 80)
    print(table_df.to_string(index=False))

    return table_df


# ------------------------------------------------------------
# Figure 4.x: CatBoost Feature Importance
# ------------------------------------------------------------

def plot_catboost_feature_importance(top_n: int = TOP_N_FEATURES):
    importance_df = read_csv_required(
        CATBOOST_IMPORTANCE_FILE,
        "CatBoost feature importance file"
    )

    require_columns(
        importance_df,
        ["feature", "importance"],
        "catboost_feature_importance.csv"
    )

    top_df = (
        importance_df
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(9, 6))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} CatBoost Feature Importance")
    plt.grid(axis="x", alpha=0.3)

    save_or_show("catboost_feature_importance")

    print("\nTop CatBoost Features:")
    print(top_df.sort_values("importance", ascending=False).to_string(index=False))

    return top_df


# ------------------------------------------------------------
# Figure 4.x: DNN Training and Validation Loss
# ------------------------------------------------------------

def plot_dnn_loss_curve():
    history_df = read_csv_required(
        DNN_HISTORY_FILE,
        "DNN training history file"
    )

    require_columns(
        history_df,
        ["loss", "val_loss"],
        "dnn_training_history.csv"
    )

    epochs = np.arange(1, len(history_df) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history_df["loss"], marker="o", label="Training Loss")
    plt.plot(epochs, history_df["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DNN Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    save_or_show("dnn_loss_curve")


# ------------------------------------------------------------
# Figure 4.x: DNN Training and Validation AUC-PR
# ------------------------------------------------------------

def plot_dnn_auc_pr_curve():
    history_df = read_csv_required(
        DNN_HISTORY_FILE,
        "DNN training history file"
    )

    require_columns(
        history_df,
        ["auc_pr", "val_auc_pr"],
        "dnn_training_history.csv"
    )

    epochs = np.arange(1, len(history_df) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history_df["auc_pr"], marker="o", label="Training AUC-PR")
    plt.plot(epochs, history_df["val_auc_pr"], marker="o", label="Validation AUC-PR")
    plt.xlabel("Epoch")
    plt.ylabel("AUC-PR")
    plt.title("DNN Training and Validation AUC-PR")
    plt.legend()
    plt.grid(alpha=0.3)

    save_or_show("dnn_auc_pr_curve")


# ------------------------------------------------------------
# Figure 4.x: Autoencoder Training and Validation Loss
# ------------------------------------------------------------

def plot_autoencoder_loss_curve():
    history_df = read_csv_required(
        AE_HISTORY_FILE,
        "Autoencoder training history file"
    )

    require_columns(
        history_df,
        ["loss", "val_loss"],
        "autoencoder_training_history.csv"
    )

    epochs = np.arange(1, len(history_df) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history_df["loss"], marker="o", label="Training Reconstruction Loss")
    plt.plot(epochs, history_df["val_loss"], marker="o", label="Validation Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Autoencoder Training and Validation Reconstruction Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    save_or_show("autoencoder_loss_curve")


# ------------------------------------------------------------
# Table 4.x: Autoencoder Reconstruction Error Summary
# ------------------------------------------------------------

def show_autoencoder_reconstruction_summary():
    summary_df = read_csv_required(
        SUMMARY_FILE,
        "Model check summary file"
    )

    ae_rows = summary_df[summary_df["model"].astype(str).str.lower().str.contains("auto")]

    if ae_rows.empty:
        raise ValueError("Autoencoder row not found in model_check_summary.csv")

    ae = ae_rows.iloc[0]

    threshold_text = read_text_optional(AE_THRESHOLD_FILE)

    ae_summary = pd.DataFrame({
        "Metric": [
            "Reconstruction threshold",
            "Mean normal reconstruction error",
            "Mean fraud reconstruction error",
            "Fraud/normal error ratio",
            "Anomaly flag rate",
            "AUC-ROC based on anomaly score",
            "AUC-PR based on anomaly score",
            "Precision based on anomaly flag",
            "Recall based on anomaly flag",
            "F1-score based on anomaly flag",
        ],
        "Value": [
            float(threshold_text) if threshold_text is not None else np.nan,
            ae.get("mean_normal_error", np.nan),
            ae.get("mean_fraud_error", np.nan),
            ae.get("fraud_normal_error_ratio", np.nan),
            ae.get("anomaly_flag_rate", np.nan),
            ae.get("auc_roc", np.nan),
            ae.get("auc_pr", np.nan),
            ae.get("precision", np.nan),
            ae.get("recall", np.nan),
            ae.get("f1", np.nan),
        ],
    })

    ae_summary["Value"] = pd.to_numeric(ae_summary["Value"], errors="coerce").round(6)

    print("\n" + "=" * 80)
    print("Table 4.x Autoencoder Reconstruction Error Summary")
    print("=" * 80)
    print(ae_summary.to_string(index=False))

    return ae_summary


# ------------------------------------------------------------
# Load prediction outputs for PR and ROC curves
# ------------------------------------------------------------

def load_prediction_outputs():
    cb_df = read_csv_required(
        CATBOOST_PRED_FILE,
        "CatBoost validation prediction file"
    )

    dnn_df = read_csv_required(
        DNN_PRED_FILE,
        "DNN validation prediction file"
    )

    ae_df = read_csv_required(
        AE_SCORE_FILE,
        "Autoencoder validation score file"
    )

    require_columns(
        cb_df,
        ["y_true", "catboost_probability"],
        "catboost_val_predictions.csv"
    )

    require_columns(
        dnn_df,
        ["y_true", "dnn_probability"],
        "dnn_val_predictions.csv"
    )

    require_columns(
        ae_df,
        ["y_true"],
        "autoencoder_val_scores.csv"
    )

    if "anomaly_score" not in ae_df.columns and "reconstruction_error" not in ae_df.columns:
        raise ValueError(
            "Autoencoder score file must contain either anomaly_score or reconstruction_error."
        )

    check_probability_range(cb_df, "catboost_probability", "CatBoost")
    check_probability_range(dnn_df, "dnn_probability", "DNN")

    y_true_cb = cb_df["y_true"].astype(int).values
    y_true_dnn = dnn_df["y_true"].astype(int).values
    y_true_ae = ae_df["y_true"].astype(int).values

    scores = {
        "CatBoost": (
            y_true_cb,
            pd.to_numeric(cb_df["catboost_probability"], errors="coerce").values,
        ),
        "DNN": (
            y_true_dnn,
            pd.to_numeric(dnn_df["dnn_probability"], errors="coerce").values,
        ),
    }

    if "anomaly_score" in ae_df.columns:
        ae_score_col = "anomaly_score"
    else:
        ae_score_col = "reconstruction_error"

    scores["Autoencoder"] = (
        y_true_ae,
        pd.to_numeric(ae_df[ae_score_col], errors="coerce").values,
    )

    return scores


# ------------------------------------------------------------
# Figure 4.x: Precision-Recall Curve Comparison Across Models
# ------------------------------------------------------------

def plot_precision_recall_comparison():
    scores = load_prediction_outputs()

    plt.figure(figsize=(8, 6))

    for model_name, (y_true, score_values) in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, score_values)
        auc_pr = average_precision_score(y_true, score_values)

        plt.plot(
            recall,
            precision,
            label=f"{model_name} (AUC-PR={auc_pr:.4f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison Across Models")
    plt.legend()
    plt.grid(alpha=0.3)

    save_or_show("precision_recall_curve_comparison")


# ------------------------------------------------------------
# Figure 4.x: ROC Curve Comparison Across Models
# ------------------------------------------------------------

def plot_roc_comparison():
    scores = load_prediction_outputs()

    plt.figure(figsize=(8, 6))

    for model_name, (y_true, score_values) in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score_values)
        auc_roc = roc_auc_score(y_true, score_values)

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC-ROC={auc_roc:.4f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison Across Models")
    plt.legend()
    plt.grid(alpha=0.3)

    save_or_show("roc_curve_comparison")


# ------------------------------------------------------------
# Short sample outputs for Chapter 4.4
# ------------------------------------------------------------

def show_sample_model_outputs():
    cb_df = read_csv_required(
        CATBOOST_PRED_FILE,
        "CatBoost validation prediction file"
    )

    dnn_df = read_csv_required(
        DNN_PRED_FILE,
        "DNN validation prediction file"
    )

    ae_df = read_csv_required(
        AE_SCORE_FILE,
        "Autoencoder validation score file"
    )

    print("\n" + "=" * 80)
    print("Short Sample Output for Chapter 4.4")
    print("=" * 80)

    cb_cols = [
        col for col in [
            "transaction_id",
            "entity_id",
            "catboost_probability",
            "y_true",
            "pred_label",
        ]
        if col in cb_df.columns
    ]

    dnn_cols = [
        col for col in [
            "transaction_id",
            "entity_id",
            "dnn_probability",
            "y_true",
            "pred_label",
        ]
        if col in dnn_df.columns
    ]

    ae_cols = [
        col for col in [
            "transaction_id",
            "entity_id",
            "reconstruction_error",
            "anomaly_score",
            "anomaly_flag",
            "y_true",
        ]
        if col in ae_df.columns
    ]

    print("\nCatBoost sample output:")
    print(cb_df[cb_cols].head(SHOW_SAMPLE_ROWS).to_string(index=False))

    print("\nDNN sample output:")
    print(dnn_df[dnn_cols].head(SHOW_SAMPLE_ROWS).to_string(index=False))

    print("\nAutoencoder sample output:")
    print(ae_df[ae_cols].head(SHOW_SAMPLE_ROWS).to_string(index=False))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\nChapter 4.3 Output Extraction and Visualization")
    print("=" * 80)

    # Tables
    show_model_performance_table()
    show_autoencoder_reconstruction_summary()

    # Main Chapter 4.3 figures
    plot_catboost_feature_importance(top_n=TOP_N_FEATURES)
    plot_dnn_loss_curve()
    plot_dnn_auc_pr_curve()
    plot_autoencoder_loss_curve()
    plot_precision_recall_comparison()
    plot_roc_comparison()

    # Short sample outputs for Chapter 4.4
    show_sample_model_outputs()

    print("\nCompleted Chapter 4.3 figure and table extraction.")


if __name__ == "__main__":
    main()