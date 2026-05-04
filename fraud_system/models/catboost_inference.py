from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import utils.config as config
from utils.schema import ModelScores


class CatBoostInferenceModel:
    """
    CatBoost inference wrapper.

    Responsibilities:
    - load trained CatBoost model
    - load CatBoost feature metadata
    - align raw input data to training schema
    - return CatBoost probability inside ModelScores
    """

    def __init__(
        self,
        model_path: Union[str, Path] = config.CATBOOST_MODEL_FILE,
        feature_list_path: Union[str, Path] = config.CATBOOST_FEATURE_LIST,
        cat_features_path: Union[str, Path] = config.CATBOOST_CAT_FEATURES,
        transaction_id_col: str = config.TRANSACTION_ID_COL,
        entity_id_col: str = config.ENTITY_ID_COL,
    ):
        self.model_path = Path(model_path)
        self.feature_list_path = Path(feature_list_path)
        self.cat_features_path = Path(cat_features_path)

        self.transaction_id_col = transaction_id_col
        self.entity_id_col = entity_id_col

        self._model: Optional[CatBoostClassifier] = None
        self._feature_cols: List[str] = []
        self._cat_features: List[str] = []

    @staticmethod
    def _load_single_column_csv(path: Path) -> List[str]:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            raise ValueError(f"CSV file is empty: {path}")
        return df.iloc[:, 0].astype(str).tolist()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call CatBoostInferenceModel.load() first.")

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"CatBoost model file not found: {self.model_path}")
        if not self.feature_list_path.exists():
            raise FileNotFoundError(f"Feature list file not found: {self.feature_list_path}")
        if not self.cat_features_path.exists():
            raise FileNotFoundError(f"Categorical feature list file not found: {self.cat_features_path}")

        self._model = CatBoostClassifier()
        self._model.load_model(str(self.model_path))

        self._feature_cols = self._load_single_column_csv(self.feature_list_path)
        self._cat_features = self._load_single_column_csv(self.cat_features_path)

        print(
            f"[CatBoost] Loaded model | "
            f"{len(self._feature_cols)} features | "
            f"{len(self._cat_features)} categorical"
        )

    def _prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_loaded()

        X = df.copy()

        for col in self._feature_cols:
            if col not in X.columns:
                if col in self._cat_features:
                    X[col] = "unknown"
                else:
                    X[col] = 0

        X = X[self._feature_cols].copy()

        for col in self._cat_features:
            if col in X.columns:
                X[col] = X[col].fillna("unknown").astype(str)

        num_cols = [c for c in self._feature_cols if c not in self._cat_features]
        if num_cols:
            X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        return X

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
        probabilities = self._model.predict_proba(X)[:, 1]

        return pd.DataFrame({
            "transaction_id": ids,
            "entity_id": entity_ids,
            "catboost_probability": probabilities,
        })

    def predict(self, df: pd.DataFrame) -> List[ModelScores]:
        """
        Main stage-level output for pipeline use.
        """
        self._ensure_loaded()

        X = self._prepare_input(df)
        probabilities = self._model.predict_proba(X)[:, 1]

        return [
            ModelScores(catboost_probability=float(prob))
            for prob in probabilities
        ]

    def predict_single(self, row: Union[pd.Series, dict]) -> ModelScores:
        df = pd.DataFrame([row])
        return self.predict(df)[0]

if __name__ == "__main__":
    print("[CatBoost Inference] Loading model...")

    model = CatBoostInferenceModel()
    model.load()

    print("[CatBoost Inference] Model loaded successfully.")
    print("[CatBoost Inference] Model path:", config.CATBOOST_MODEL_FILE)
    print("[CatBoost Inference] Feature count:", len(model._feature_cols))
    print("[CatBoost Inference] Categorical feature count:", len(model._cat_features))