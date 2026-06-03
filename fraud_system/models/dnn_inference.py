import pickle
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

import numpy as np
import pandas as pd
from tensorflow import keras

import utils.config as config
from utils.schema import ModelScores


class DNNInferenceModel:
    """
    DNN inference wrapper.

    Responsibilities:
    - load trained Keras DNN model
    - load Step 11B preprocessing artifacts
    - apply the same inference-time preprocessing
    - return DNN probability inside ModelScores
    """

    def __init__(
        self,
        model_path: Union[str, Path] = config.DNN_MODEL_FILE,
        feature_list_path: Union[str, Path] = config.DNN_FEATURE_LIST,
        scaler_path: Union[str, Path] = config.DNN_SCALER_PKL,
        freq_maps_path: Union[str, Path] = config.DNN_FREQ_MAPS_PKL,
        sentinel_path: Union[str, Path] = config.DNN_SENTINEL_PKL,
        transaction_id_col: str = config.TRANSACTION_ID_COL,
        entity_id_col: str = config.ENTITY_ID_COL,
    ):
        self.model_path = Path(model_path)
        self.feature_list_path = Path(feature_list_path)
        self.scaler_path = Path(scaler_path)
        self.freq_maps_path = Path(freq_maps_path)
        self.sentinel_path = Path(sentinel_path)

        self.transaction_id_col = transaction_id_col
        self.entity_id_col = entity_id_col

        self._model: Optional[keras.Model] = None
        self._feature_cols: List[str] = []
        self._scaler = None
        self._freq_maps: Dict[str, Dict[Any, float]] = {}
        self._sentinels: Dict[str, float] = {}

    @staticmethod
    def _load_single_column_csv(path: Path) -> List[str]:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            raise ValueError(f"CSV file is empty: {path}")
        return df.iloc[:, 0].astype(str).tolist()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call DNNInferenceModel.load() first.")

    def load(self) -> None:
        required = [
            self.model_path,
            self.feature_list_path,
            self.scaler_path,
            self.freq_maps_path,
            self.sentinel_path,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required DNN artifact not found: {path}")

        self._model = keras.models.load_model(str(self.model_path))
        self._feature_cols = self._load_single_column_csv(self.feature_list_path)

        with open(self.scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        with open(self.freq_maps_path, "rb") as f:
            self._freq_maps = pickle.load(f)

        with open(self.sentinel_path, "rb") as f:
            self._sentinels = pickle.load(f)

        print(
            f"[DNN] Loaded model | "
            f"{len(self._feature_cols)} features | "
            f"{len(self._freq_maps)} freq-encoded columns | "
            f"{len(self._sentinels)} sentinel replacements"
        )


    def _prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_loaded()

        # # --- DEBUG START ---
        # print(f"[DNN DEBUG] Input df columns[:5]: {df.columns[:5].tolist()}")
        # print(f"[DNN DEBUG] Expected feature_cols[:5]: {self._feature_cols[:5]}")
        # print(f"[DNN DEBUG] Input df shape: {df.shape}")
        # # --- DEBUG END ---

        X = df.copy()

        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = 0

        X = X[self._feature_cols].copy()

        # Step 1: frequency encoding
        for col, freq_map in self._freq_maps.items():
            if col in X.columns:
                X[col] = X[col].map(freq_map).fillna(0).astype(np.float32)

        # Step 2: sentinel replacement
        for col, replacement in self._sentinels.items():
            if col in X.columns:
                X[col] = X[col].replace(-999, replacement)

        # Step 3: numeric casting
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Step 4: scaling
        # X_scaled = self._scaler.transform(X).astype(np.float32)

        # # --- DEBUG: after scaling ---
        X_scaled = self._scaler.transform(X).astype(np.float32)
        X_scaled = np.clip(X_scaled, -5, 5)
        # print(f"[DNN DEBUG] Post-scale global min: {X_scaled.min():.4f}")
        # print(f"[DNN DEBUG] Post-scale global max: {X_scaled.max():.4f}")
        # print(f"[DNN DEBUG] Post-scale mean: {X_scaled.mean():.4f}")
        # # --- DEBUG END ---

        return X_scaled

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Debug-friendly probability output as DataFrame.
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

        X = self._prepare_input(df)
        probabilities = self._model.predict(X, verbose=0).flatten()

        return pd.DataFrame({
            "transaction_id": ids,
            "entity_id": entity_ids,
            "dnn_probability": probabilities.astype(np.float32),
        })

    def predict(self, df: pd.DataFrame) -> List[ModelScores]:
        """
        Main stage-level output for pipeline use.
        """
        self._ensure_loaded()

        X = self._prepare_input(df)
        probabilities = self._model.predict(X, verbose=0).flatten()

        return [
            ModelScores(dnn_probability=float(prob))
            for prob in probabilities
        ]

    def predict_single(
        self,
        row: Union[pd.Series, dict],
        existing_scores: Optional[ModelScores] = None,
    ) -> ModelScores:
        df = pd.DataFrame([row])
        score = float(self._model.predict(self._prepare_input(df), verbose=0).flatten()[0])

        if existing_scores is not None:
            existing_scores.dnn_probability = score
            return existing_scores

        return ModelScores(dnn_probability=score)


if __name__ == "__main__":
    print("[DNN Inference] Loading model and artifacts...")

    model = DNNInferenceModel()
    model.load()

    print("[DNN Inference] Model loaded successfully.")
    print("[DNN Inference] Model path:", config.DNN_MODEL_FILE)
    print("[DNN Inference] Feature artifact:", config.DNN_FEATURE_LIST)
    print("[DNN Inference] Scaler artifact:", config.DNN_SCALER_PKL)