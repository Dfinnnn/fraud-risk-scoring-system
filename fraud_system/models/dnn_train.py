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


class DNNTrainer:
    """
    Train, evaluate, and save the DNN fraud model.

    Responsibilities:
    - load DNN-ready training/validation datasets
    - build DNN architecture
    - train with early stopping and LR scheduling
    - evaluate validation performance
    - save trained model and training history
    """

    def __init__(
        self,
        train_file: Path = config.DNN_TRAIN_FILE,
        val_file: Path = config.DNN_VAL_FILE,
        feature_list_file: Path = config.DNN_FEATURE_LIST,
        model_output_file: Path = config.DNN_MODEL_FILE,
        history_output_file: Path = config.DNN_TRAINING_HISTORY_FILE,
        target_col: str = config.TARGET_COL,
        transaction_id_col: str = config.TRANSACTION_ID_COL,
        random_seed: int = config.RANDOM_SEED,
    ):
        self.train_file = Path(train_file)
        self.val_file = Path(val_file)
        self.feature_list_file = Path(feature_list_file)
        self.model_output_file = Path(model_output_file)
        self.history_output_file = Path(history_output_file)
        self.target_col = target_col
        self.transaction_id_col = transaction_id_col
        self.random_seed = random_seed

        self.feature_cols: List[str] = []
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None

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

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        missing_features = [c for c in self.feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(
                f"Dataset missing required DNN features: {missing_features[:10]}"
                + (" ..." if len(missing_features) > 10 else "")
            )

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        drop_cols = [self.target_col]
        if self.transaction_id_col in df.columns:
            drop_cols.append(self.transaction_id_col)

        X = df.drop(columns=drop_cols)[self.feature_cols].values.astype(np.float32)
        y = df[self.target_col].values.astype(np.float32)

        return X, y

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._validate_files()

        self.feature_cols = self._load_single_column_csv(self.feature_list_file)

        train_df = pd.read_csv(self.train_file)
        val_df = pd.read_csv(self.val_file)

        X_train, y_train = self._prepare_xy(train_df)
        X_val, y_val = self._prepare_xy(val_df)

        return X_train, y_train, X_val, y_val

    def build_model(self, input_dim: int) -> keras.Model:
        """
        V1 architecture:
        input -> 128 -> 64 -> 32 -> sigmoid

        Reason:
        - strong enough for behavioural tabular interactions
        - simpler and safer than a larger network for first validation
        - easier to tune if overfitting appears
        """
        inputs = keras.Input(shape=(input_dim,), name="input")

        x = layers.Dense(128, name="dense_1")(inputs)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Activation("relu", name="relu_1")(x)
        x = layers.Dropout(0.30, name="dropout_1")(x)

        x = layers.Dense(64, name="dense_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Activation("relu", name="relu_2")(x)
        x = layers.Dropout(0.30, name="dropout_2")(x)

        x = layers.Dense(32, name="dense_3")(x)
        x = layers.Activation("relu", name="relu_3")(x)
        x = layers.Dropout(0.20, name="dropout_3")(x)

        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="fraud_dnn")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.AUC(name="auc_roc"),
                keras.metrics.AUC(name="auc_pr", curve="PR"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        self.model = model
        return model

    def train(self) -> Dict[str, Any]:
        X_train, y_train, X_val, y_val = self.load_data()

        if self.model is None:
            self.build_model(input_dim=X_train.shape[1])

        early_stop = callbacks.EarlyStopping(
            monitor="val_auc_pr",
            patience=10,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr",
            factor=0.5,
            patience=5,
            mode="max",
            min_lr=1e-6,
            verbose=1,
        )

        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(self.model_output_file),
            monitor="val_auc_pr",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=2048,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1,
        )

        best_model = keras.models.load_model(str(self.model_output_file))

        val_probs = best_model.predict(X_val, verbose=0).flatten()
        val_preds = (val_probs >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average="binary", zero_division=0
        )

        metrics = {
            "auc_roc": float(roc_auc_score(y_val, val_probs)),
            "auc_pr": float(average_precision_score(y_val, val_probs)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "epochs_ran": int(len(self.history.history["loss"])),
            "best_val_auc_pr": float(max(self.history.history["val_auc_pr"])),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "feature_count": int(X_train.shape[1]),
        }

        return metrics

    def save_history(self) -> None:
        if self.history is None:
            raise RuntimeError("Model has not been trained yet.")

        history_df = pd.DataFrame(self.history.history)
        self.history_output_file.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(self.history_output_file, index=False)

    def run(self) -> Dict[str, Any]:
        metrics = self.train()
        self.save_history()
        return metrics


if __name__ == "__main__":
    print("[DNN Train] Starting training...")

    trainer = DNNTrainer()
    metrics = trainer.run()

    print("[DNN Train] Training completed successfully.")
    print("[DNN Train] Model saved to:", config.DNN_MODEL_FILE)
    print("[DNN Train] History saved to:", config.DNN_TRAINING_HISTORY_FILE)
    print("[DNN Train] Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")