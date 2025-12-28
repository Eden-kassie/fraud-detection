"""Baseline model implementations."""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from src.utils.logger import logger
from src.utils.config import LOGISTIC_REGRESSION_PARAMS
from src.models.evaluation import calculate_metrics, print_classification_report


class LogisticRegressionBaseline:
    """Logistic Regression baseline model for fraud detection."""

    def __init__(self, **kwargs):
        """
        Initialize Logistic Regression model.

        Args:
            **kwargs: Additional parameters for LogisticRegression
        """
        params = {**LOGISTIC_REGRESSION_PARAMS, **kwargs}
        self.model = LogisticRegression(**params)
        self.model_name = "Logistic Regression"
        logger.info(f"Initialized {self.model_name} with params: {params}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Predicted probabilities for positive class
        """
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {self.model_name}...")

        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['model_name'] = self.model_name  # Add for comparison

        logger.info(f"\n{self.model_name} Metrics:")
        for metric, value in metrics.items():
            if metric != 'model_name':  # Skip non-numeric
                logger.info(f"  {metric}: {value:.4f}")

        print_classification_report(y_test, y_pred, self.model_name)

        return metrics

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature coefficients.

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature coefficients
        """
        coefficients = self.model.coef_[0]
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)

        return feat_imp
