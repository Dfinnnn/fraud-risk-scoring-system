from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

import utils.config as config
from utils.schema import ModelScores


@dataclass
class AnomalyResult:
    reconstruction_error: float
    anomaly_score: float
    anomaly_flag: bool

    def to_dict(self) -> dict:
        return {
            "reconstruction_error": self.reconstruction_error,
            "anomaly_score": self.anomaly_score,
            "anomaly_flag": self.anomaly_flag,
        }


class AutoencoderInferenceModel:
    """
    Autoencoder inference wrapper.

    Responsibilities:
    - load trained autoencoder model
    - load Step 11C preprocessing artifacts
    - apply the same preprocessing used in AE training
    - compute reconstruction error
    - derive anomaly_score and anomaly_flag
    """

    def __init__(
        self,
        model_path: Union[str, Path] = config.AE_MODEL_FILE,
        feature_list_path: Union[str, Path] = config.AE_FEATURE_LIST,
        scaler_path: Union[str, Path] = config.AE_SCALER_PKL,
        freq_maps_path: Union[str, Path] = config.AE_FREQ_MAPS_PKL,
        sentinel_path: Union[str, Path] = config.AE_SENTINEL_PKL,
        threshold_path: Union[str, Path] = config.AE_THRESHOLD_FILE,
        transaction_id_col: str = config.TRANSACTION_ID_COL,
        entity_id_col: str = config.ENTITY_ID_COL,
    ):
        self.model_path = Path(model_path)
        self.feature_list_path = Path(feature_list_path)
        self.scaler_path = Path(scaler_path)
        self.freq_maps_path = Path(freq_maps_path)
        self.sentinel_path = Path(sentinel_path)
        self.threshold_path = Path(threshold_path)

        self.transaction_id_col = transaction_id_col
        self.entity_id_col = entity_id_col

        self._model: Optional[keras.Model] = None
        self._feature_cols: List[str] = []
        self._scaler = None
        self._freq_maps: Dict[str, Dict[Any, float]] = {}
        self._sentinels: Dict[str, float] = {}
        self._threshold: Optional[float] = None

    @staticmethod
    def _load_single_column_csv(path: Path) -> List[str]:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            raise ValueError(f"CSV file is empty: {path}")
        return df.iloc[:, 0].astype(str).tolist()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call AutoencoderInferenceModel.load() first.")

    def load(self) -> None:
        required = [
            self.model_path,
            self.feature_list_path,
            self.scaler_path,
            self.freq_maps_path,
            self.sentinel_path,
            self.threshold_path,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required AE artifact not found: {path}")

        self._model = keras.models.load_model(str(self.model_path))
        self._feature_cols = self._load_single_column_csv(self.feature_list_path)

        with open(self.scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        with open(self.freq_maps_path, "rb") as f:
            self._freq_maps = pickle.load(f)

        with open(self.sentinel_path, "rb") as f:
            self._sentinels = pickle.load(f)

        self._threshold = float(self.threshold_path.read_text().strip())

        print(
            f"[AE] Loaded model | "
            f"{len(self._feature_cols)} features | "
            f"{len(self._freq_maps)} freq-encoded columns | "
            f"{len(self._sentinels)} sentinel replacements | "
            f"threshold={self._threshold:.6f}"
        )

    def _prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the same preprocessing used during Step 11C / AE training.

        Steps:
        1. align to feature list
        2. frequency-encode categoricals
        3. replace -999 sentinels
        4. numeric casting
        5. standard scaling
        """
        self._ensure_loaded()

        # # --- DEBUG START ---
        # print(f"[AE DEBUG] Input df columns[:5]: {df.columns[:5].tolist()}")
        # print(f"[AE DEBUG] Expected feature_cols[:5]: {self._feature_cols[:5]}")
        # print(f"[AE DEBUG] Input df shape: {df.shape}")
        # # --- DEBUG END ---

        X = df.copy()

        # Add missing columns safely
        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = 0

        X = X[self._feature_cols].copy()

        # Step 1: frequency encode categoricals
        for col, freq_map in self._freq_maps.items():
            if col in X.columns:
                X[col] = X[col].map(freq_map).fillna(0).astype(np.float32)

        # Step 2: replace -999 sentinels
        for col, replacement in self._sentinels.items():
            if col in X.columns:
                X[col] = X[col].replace(-999, replacement)

        # Step 3: numeric casting
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # # Step 4: scaling
        X_scaled = self._scaler.transform(X).astype(np.float32)
        X_scaled = np.clip(X_scaled, -5, 5)
        # print(f"[AE DEBUG] Post-scale global min: {X_scaled.min():.4f}")
        # print(f"[AE DEBUG] Post-scale global max: {X_scaled.max():.4f}")
        # print(f"[AE DEBUG] Post-scale mean: {X_scaled.mean():.4f}")
        # # --- DEBUG END ---

        return X_scaled

    @staticmethod
    def _row_reconstruction_error(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        return np.mean(np.square(x_true - x_pred), axis=1)

    def score(self, df: pd.DataFrame) -> List[AnomalyResult]:
        """
        Main batch anomaly scoring function.
        """
        self._ensure_loaded()

        X = self._prepare_input(df)
        X_reconstructed = self._model.predict(X, verbose=0)

        errors = self._row_reconstruction_error(X, X_reconstructed)

        results: List[AnomalyResult] = []
        for err in errors:
            anomaly_score = float(err) / float(self._threshold)
            results.append(
                AnomalyResult(
                    reconstruction_error=float(err),
                    anomaly_score=anomaly_score,
                    anomaly_flag=anomaly_score > 1.0,
                )
            )

        return results

    def score_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Debug-friendly DataFrame output for inspection.
        """
        self._ensure_loaded()

        ids = (
            df[self.transaction_id_col].astype(str).values
            if self.transaction_id_col in df.columns
            else np.arange(len(df)).astype(str)
        )

        entity_ids = (
            df[self.entity_id_col].astype(str).values
            if self.entity_id_col in df.columns
            else np.array(["unknown_entity"] * len(df))
        )

        results = self.score(df)

        return pd.DataFrame({
            "transaction_id": ids,
            "entity_id": entity_ids,
            "reconstruction_error": [r.reconstruction_error for r in results],
            "anomaly_score": [r.anomaly_score for r in results],
            "anomaly_flag": [r.anomaly_flag for r in results],
        })

    def score_single(
        self,
        row: Union[pd.Series, dict],
        existing_scores: Optional[ModelScores] = None,
    ) -> ModelScores:
        """
        Score one transaction and optionally merge into an existing ModelScores object.
        """
        df = pd.DataFrame([row])
        result = self.score(df)[0]

        if existing_scores is not None:
            existing_scores.reconstruction_error = result.reconstruction_error
            existing_scores.anomaly_score = result.anomaly_score
            existing_scores.anomaly_flag = result.anomaly_flag
            return existing_scores

        return ModelScores(
            reconstruction_error=result.reconstruction_error,
            anomaly_score=result.anomaly_score,
            anomaly_flag=result.anomaly_flag,
        )


if __name__ == "__main__":
    print("[Autoencoder Inference] Loading model and artifacts...")

    model = AutoencoderInferenceModel()
    model.load()

    print("[Autoencoder Inference] Model loaded successfully.")
    print("[Autoencoder Inference] Model path:", config.AE_MODEL_FILE)
    print("[Autoencoder Inference] Threshold path:", config.AE_THRESHOLD_FILE)
    print("[Autoencoder Inference] Feature artifact:", config.AE_FEATURE_LIST)
    print("[Autoencoder Inference] Scaler artifact:", config.AE_SCALER_PKL)