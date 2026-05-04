from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

import utils.config as config


class CatBoostTrainer:
    """
    Train, evaluate, and save the CatBoost fraud model.

    Responsibilities:
    - load training/validation datasets
    - load feature metadata and optional class weights
    - train CatBoost using raw categorical features
    - evaluate validation performance
    - save trained model and feature importance
    """

    def __init__(
        self,
        train_file: Path = config.CATBOOST_TRAIN_FILE,
        val_file: Path = config.CATBOOST_VAL_FILE,
        feature_list_file: Path = config.CATBOOST_FEATURE_LIST,
        cat_features_file: Path = config.CATBOOST_CAT_FEATURES,
        class_weights_file: Path = config.CATBOOST_CLASS_WEIGHTS,
        model_output_file: Path = config.CATBOOST_MODEL_FILE,
        importance_output_file: Path = config.CATBOOST_FEATURE_IMPORTANCE_FILE,
        target_col: str = config.TARGET_COL,
    ):
        self.train_file = Path(train_file)
        self.val_file = Path(val_file)
        self.feature_list_file = Path(feature_list_file)
        self.cat_features_file = Path(cat_features_file)
        self.class_weights_file = Path(class_weights_file)
        self.model_output_file = Path(model_output_file)
        self.importance_output_file = Path(importance_output_file)
        self.target_col = target_col

        self.feature_cols: List[str] = []
        self.cat_features: List[str] = []
        self.class_weights: Optional[Dict[Any, float]] = None
        self.model: Optional[CatBoostClassifier] = None

    @staticmethod
    def _load_single_column_csv(path: Path) -> List[str]:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            raise ValueError(f"CSV file is empty: {path}")
        return df.iloc[:, 0].astype(str).tolist()

    def _load_class_weights(self) -> Optional[Dict[Any, float]]:
        if not self.class_weights_file.exists():
            return None

        cw_df = pd.read_csv(self.class_weights_file)
        if cw_df.shape[1] < 2:
            raise ValueError(
                f"Class weight file must contain at least two columns: {self.class_weights_file}"
            )

        return dict(zip(cw_df.iloc[:, 0], cw_df.iloc[:, 1]))

    def _validate_files(self) -> None:
        required = [
            self.train_file,
            self.val_file,
            self.feature_list_file,
            self.cat_features_file,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        missing_features = [c for c in self.feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(
                f"Dataset missing required CatBoost features: {missing_features[:10]}"
                + (" ..." if len(missing_features) > 10 else "")
            )

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()

        for col in self.cat_features:
            if col in X.columns:
                X[col] = X[col].fillna("unknown").astype(str)

        num_cols = [c for c in self.feature_cols if c not in self.cat_features]
        if num_cols:
            X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        return X, y

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._validate_files()

        self.feature_cols = self._load_single_column_csv(self.feature_list_file)
        self.cat_features = self._load_single_column_csv(self.cat_features_file)
        self.class_weights = self._load_class_weights()

        train_df = pd.read_csv(self.train_file)
        val_df = pd.read_csv(self.val_file)

        return train_df, val_df

    def build_model(self) -> CatBoostClassifier:
        model_kwargs = dict(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=config.RANDOM_SEED,
            early_stopping_rounds=50,
            verbose=100,
        )

        if self.class_weights is not None:
            model_kwargs["class_weights"] = self.class_weights
        else:
            model_kwargs["auto_class_weights"] = "Balanced"

        self.model = CatBoostClassifier(**model_kwargs)
        return self.model

    def train(self) -> Dict[str, Any]:
        train_df, val_df = self.load_data()

        X_train, y_train = self._prepare_xy(train_df)
        X_val, y_val = self._prepare_xy(val_df)

        if self.model is None:
            self.build_model()

        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        val_pool = Pool(X_val, y_val, cat_features=self.cat_features)

        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )

        val_probs = self.model.predict_proba(val_pool)[:, 1]
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
            "best_iteration": int(self.model.best_iteration_) if self.model.best_iteration_ is not None else None,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "feature_count": int(len(self.feature_cols)),
            "cat_feature_count": int(len(self.cat_features)),
        }

        return metrics

    def save_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        self.model_output_file.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.model_output_file))

    def save_feature_importance(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.get_feature_importance(),
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        self.importance_output_file.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(self.importance_output_file, index=False)

    def run(self) -> Dict[str, Any]:
        metrics = self.train()
        self.save_model()
        self.save_feature_importance()
        return metrics

if __name__ == "__main__":
    print("[CatBoost Train] Starting training...")

    trainer = CatBoostTrainer()
    metrics = trainer.run()

    print("[CatBoost Train] Training completed successfully.")
    print("[CatBoost Train] Model saved to:", config.CATBOOST_MODEL_FILE)
    print("[CatBoost Train] Feature importance saved to:", config.CATBOOST_FEATURE_IMPORTANCE_FILE)
    print("[CatBoost Train] Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")