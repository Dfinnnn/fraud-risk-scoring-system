from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

import utils.config as config


class AutoencoderTrainer:
    """
    Train, calibrate, evaluate, and save the autoencoder anomaly model.

    Responsibilities:
    - load AE-ready datasets and feature metadata
    - train only on normal transactions
    - calibrate threshold on validation-normal reconstruction errors
    - verify separation between normal and fraud reconstruction errors
    - save model, threshold, and training history
    """

    def __init__(
        self,
        train_file: Path = config.AE_TRAIN_FILE,
        val_file: Path = config.AE_VAL_FILE,
        feature_list_file: Path = config.AE_FEATURE_LIST,
        model_output_file: Path = config.AE_MODEL_FILE,
        threshold_output_file: Path = config.AE_THRESHOLD_FILE,
        history_output_file: Path = config.AE_TRAINING_HISTORY_FILE,
        target_col: str = config.TARGET_COL,
        transaction_id_col: str = config.TRANSACTION_ID_COL,
        random_seed: int = config.RANDOM_SEED,
        threshold_quantile: float = config.AE_THRESHOLD_QUANTILE,
    ):
        self.train_file = Path(train_file)
        self.val_file = Path(val_file)
        self.feature_list_file = Path(feature_list_file)
        self.model_output_file = Path(model_output_file)
        self.threshold_output_file = Path(threshold_output_file)
        self.history_output_file = Path(history_output_file)

        self.target_col = target_col
        self.transaction_id_col = transaction_id_col
        self.random_seed = random_seed
        self.threshold_quantile = threshold_quantile

        self.feature_cols: List[str] = []
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None
        self.threshold: Optional[float] = None

        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

    @staticmethod
    def _load_single_column_csv(path: Path) -> List[str]:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            raise ValueError(f"CSV file is empty: {path}")
        return df.iloc[:, 0].astype(str).tolist()

    def _validate_files(self) -> None:
        required = [
            self.train_file,
            self.val_file,
            self.feature_list_file,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

    def _prepare_x(self, df: pd.DataFrame) -> np.ndarray:
        missing_features = [c for c in self.feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(
                f"Dataset missing required AE features: {missing_features[:10]}"
                + (" ..." if len(missing_features) > 10 else "")
            )

        drop_cols = []
        if self.transaction_id_col in df.columns:
            drop_cols.append(self.transaction_id_col)
        if self.target_col in df.columns:
            drop_cols.append(self.target_col)

        X = df.drop(columns=drop_cols, errors="ignore")[self.feature_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        return X.values.astype(np.float32)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
        - X_train_normal : normal-only training matrix
        - X_val_normal   : normal-only validation matrix
        - X_val_full     : full validation matrix
        - y_val_full     : full validation labels
        """
        self._validate_files()
        self.feature_cols = self._load_single_column_csv(self.feature_list_file)

        train_df = pd.read_csv(self.train_file)
        val_df = pd.read_csv(self.val_file)

        if self.target_col in train_df.columns:
            fraud_count = int(train_df[self.target_col].sum())
            if fraud_count != 0:
                raise ValueError(
                    f"AE training file must contain only normal rows, but found {fraud_count} fraud rows."
                )

        if self.target_col not in val_df.columns:
            raise ValueError(f"Validation file must contain '{self.target_col}' for AE evaluation.")

        X_train_normal = self._prepare_x(train_df)

        val_normal_df = val_df[val_df[self.target_col] == 0].copy()
        if len(val_normal_df) == 0:
            raise ValueError("No normal rows found in AE validation file.")

        X_val_normal = self._prepare_x(val_normal_df)
        X_val_full = self._prepare_x(val_df)
        y_val_full = val_df[self.target_col].values.astype(np.int32)

        return X_train_normal, X_val_normal, X_val_full, y_val_full

    def build_model(self, input_dim: int) -> keras.Model:
        """
        V1 undercomplete symmetric autoencoder:
        input -> 128 -> 64 -> 32 -> 64 -> 128 -> reconstruction

        Reason:
        - stable and simple
        - enough compression to model normal behaviour
        - not too expressive, so anomalies should reconstruct worse
        """
        inputs = keras.Input(shape=(input_dim,), name="ae_input")

        # Encoder
        x = layers.Dense(128, name="enc_dense_1")(inputs)
        x = layers.BatchNormalization(name="enc_bn_1")(x)
        x = layers.Activation("relu", name="enc_relu_1")(x)

        x = layers.Dense(64, name="enc_dense_2")(x)
        x = layers.BatchNormalization(name="enc_bn_2")(x)
        x = layers.Activation("relu", name="enc_relu_2")(x)

        # Bottleneck
        x = layers.Dense(32, activation="relu", name="bottleneck")(x)

        # Decoder
        x = layers.Dense(64, name="dec_dense_1")(x)
        x = layers.BatchNormalization(name="dec_bn_1")(x)
        x = layers.Activation("relu", name="dec_relu_1")(x)

        x = layers.Dense(128, name="dec_dense_2")(x)
        x = layers.BatchNormalization(name="dec_bn_2")(x)
        x = layers.Activation("relu", name="dec_relu_2")(x)

        # Reconstruction
        outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="fraud_autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )

        self.model = model
        return model

    @staticmethod
    def _row_reconstruction_error(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.square(x_true - x_pred), axis=1)

    def _calibrate_threshold(self, normal_errors: np.ndarray) -> float:
        return float(np.quantile(normal_errors, self.threshold_quantile))

    def train(self) -> Dict[str, Any]:
        X_train_normal, X_val_normal, X_val_full, y_val_full = self.load_data()

        if self.model is None:
            self.build_model(input_dim=X_train_normal.shape[1])

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            mode="min",
            min_lr=1e-6,
            verbose=1,
        )

        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(self.model_output_file),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        )

        self.history = self.model.fit(
            X_train_normal,
            X_train_normal,
            validation_data=(X_val_normal, X_val_normal),
            epochs=100,
            batch_size=1024,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1,
        )

        best_model = keras.models.load_model(str(self.model_output_file))

        # Threshold calibration using normal validation rows only
        val_normal_pred = best_model.predict(X_val_normal, verbose=0)
        val_normal_errors = self._row_reconstruction_error(X_val_normal, val_normal_pred)
        self.threshold = self._calibrate_threshold(val_normal_errors)

        # Full validation evaluation
        val_full_pred = best_model.predict(X_val_full, verbose=0)
        val_full_errors = self._row_reconstruction_error(X_val_full, val_full_pred)
        anomaly_flags = (val_full_errors >= self.threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_full, anomaly_flags, average="binary", zero_division=0
        )

        # Separation diagnostics
        fraud_mask = (y_val_full == 1)
        normal_mask = (y_val_full == 0)

        fraud_errors = val_full_errors[fraud_mask]
        normal_errors_full = val_full_errors[normal_mask]

        fraud_caught = int(anomaly_flags[fraud_mask].sum()) if fraud_mask.sum() > 0 else 0
        fraud_total = int(fraud_mask.sum())
        false_alarms = int(anomaly_flags[normal_mask].sum()) if normal_mask.sum() > 0 else 0
        normal_total = int(normal_mask.sum())

        metrics = {
            "auc_roc": float(roc_auc_score(y_val_full, val_full_errors)),
            "auc_pr": float(average_precision_score(y_val_full, val_full_errors)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(self.threshold),
            "threshold_quantile": float(self.threshold_quantile),
            "epochs_ran": int(len(self.history.history["loss"])),
            "best_val_loss": float(min(self.history.history["val_loss"])),
            "train_normal_rows": int(len(X_train_normal)),
            "val_normal_rows": int(len(X_val_normal)),
            "val_full_rows": int(len(X_val_full)),
            "feature_count": int(X_train_normal.shape[1]),
            "mean_val_normal_error": float(np.mean(val_normal_errors)),
            "std_val_normal_error": float(np.std(val_normal_errors)),
            "mean_val_full_normal_error": float(np.mean(normal_errors_full)) if normal_total > 0 else None,
            "mean_val_full_fraud_error": float(np.mean(fraud_errors)) if fraud_total > 0 else None,
            "fraud_normal_error_ratio": (
                float(np.mean(fraud_errors) / np.mean(normal_errors_full))
                if fraud_total > 0 and normal_total > 0 and np.mean(normal_errors_full) > 0
                else None
            ),
            "fraud_caught": fraud_caught,
            "fraud_total": fraud_total,
            "fraud_catch_rate": float(fraud_caught / fraud_total) if fraud_total > 0 else None,
            "false_alarms": false_alarms,
            "normal_total": normal_total,
            "false_alarm_rate": float(false_alarms / normal_total) if normal_total > 0 else None,
        }

        return metrics

    def save_threshold(self) -> None:
        if self.threshold is None:
            raise RuntimeError("Threshold has not been calibrated yet.")
        self.threshold_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.threshold_output_file, "w") as f:
            f.write(str(self.threshold))

    def save_history(self) -> None:
        if self.history is None:
            raise RuntimeError("Model has not been trained yet.")
        history_df = pd.DataFrame(self.history.history)
        self.history_output_file.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(self.history_output_file, index=False)

    def run(self) -> Dict[str, Any]:
        metrics = self.train()
        self.save_threshold()
        self.save_history()
        return metrics


if __name__ == "__main__":
    print("[Autoencoder Train] Starting training...")

    trainer = AutoencoderTrainer()
    metrics = trainer.run()

    print("[Autoencoder Train] Training completed successfully.")
    print("[Autoencoder Train] Model saved to:", config.AE_MODEL_FILE)
    print("[Autoencoder Train] Threshold saved to:", config.AE_THRESHOLD_FILE)
    print("[Autoencoder Train] History saved to:", config.AE_TRAINING_HISTORY_FILE)
    print("[Autoencoder Train] Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")