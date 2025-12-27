"""Ensemble model implementations."""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.utils.logger import logger
from src.utils.config import (
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    RANDOM_STATE
)
from src.models.evaluation import calculate_metrics, print_classification_report


class RandomForestModel:
    """Random Forest classifier for fraud detection."""

    def __init__(self, **kwargs):
        """Initialize Random Forest model."""
        params = {**RANDOM_FOREST_PARAMS, **kwargs}
        self.model = RandomForestClassifier(**params)
        self.model_name = "Random Forest"
        logger.info(f"Initialized {self.model_name} with params: {params}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model."""
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model."""
        logger.info(f"Evaluating {self.model_name}...")

        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        logger.info(f"\n{self.model_name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        print_classification_report(y_test, y_pred, self.model_name)

        return metrics

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importances."""
        importances = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feat_imp


class XGBoostModel:
    """XGBoost classifier for fraud detection."""

    def __init__(self, **kwargs):
        """Initialize XGBoost model."""
        params = {**XGBOOST_PARAMS, **kwargs}
        self.model = XGBClassifier(**params)
        self.model_name = "XGBoost"
        logger.info(f"Initialized {self.model_name} with params: {params}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model."""
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model."""
        logger.info(f"Evaluating {self.model_name}...")

        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        logger.info(f"\n{self.model_name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        print_classification_report(y_test, y_pred, self.model_name)

        return metrics

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importances."""
        importances = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feat_imp


class LightGBMModel:
    """LightGBM classifier for fraud detection."""

    def __init__(self, **kwargs):
        """Initialize LightGBM model."""
        params = {**LIGHTGBM_PARAMS, **kwargs}
        self.model = LGBMClassifier(**params)
        self.model_name = "LightGBM"
        logger.info(f"Initialized {self.model_name} with params: {params}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model."""
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model."""
        logger.info(f"Evaluating {self.model_name}...")

        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        logger.info(f"\n{self.model_name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        print_classification_report(y_test, y_pred, self.model_name)

        return metrics

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importances."""
        importances = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feat_imp


def tune_hyperparameters(
    model_class: type,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict,
    search_type: str = "grid",
    cv: int = 3,
    scoring: str = "average_precision"
) -> tuple:
    """
    Perform hyperparameter tuning.

    Args:
        model_class: Model class to tune
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for search
        search_type: 'grid' or 'random'
        cv: Number of cross-validation folds
        scoring: Scoring metric

    Returns:
        Tuple of (best_model, best_params, best_score)
    """
    logger.info(f"Performing {search_type} search for hyperparameter tuning...")

    base_model = model_class()

    if search_type == "grid":
        search = GridSearchCV(
            base_model.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
    else:  # random
        search = RandomizedSearchCV(
            base_model.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            n_iter=10,
            random_state=RANDOM_STATE
        )

    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best {scoring} score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_
