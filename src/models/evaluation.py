"""Model evaluation utilities."""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import joblib
from pathlib import Path
from src.utils.logger import logger
from src.utils.config import CV_FOLDS, RANDOM_STATE, MODELS_DIR


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)

    return metrics


def stratified_kfold_cv(
    model: any,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    scoring: Optional[Dict[str, str]] = None
) -> Dict[str, np.ndarray]:
    """
    Perform stratified K-fold cross-validation.

    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target variable
        n_folds: Number of folds
        random_state: Random state
        scoring: Dictionary of scoring metrics

    Returns:
        Dictionary of cross-validation scores
    """
    logger.info(f"Performing {n_folds}-fold stratified cross-validation...")

    if scoring is None:
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'auc_pr': 'average_precision'
        }

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    return cv_results


def aggregate_cv_results(cv_results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Aggregate cross-validation results with mean and std.

    Args:
        cv_results: Cross-validation results from stratified_kfold_cv

    Returns:
        DataFrame with aggregated metrics
    """
    metrics_summary = []

    for key, values in cv_results.items():
        if key.startswith('test_'):
            metric_name = key.replace('test_', '')
            metrics_summary.append({
                'metric': metric_name,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            })

    summary_df = pd.DataFrame(metrics_summary)

    logger.info("\nCross-Validation Results:")
    logger.info(f"\n{summary_df.to_string(index=False)}")

    return summary_df


def save_model(model: any, model_name: str, output_dir: Optional[Path] = None):
    """
    Save trained model to disk.

    Args:
        model: Trained model
        model_name: Name for the saved model file
        output_dir: Optional custom output directory
    """
    output_path = output_dir or MODELS_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    if not model_name.endswith('.joblib'):
        model_name = f"{model_name}.joblib"

    filepath = output_path / model_name
    joblib.dump(model, filepath)
    logger.info(f"Saved model to {filepath}")


def load_model(model_name: str, models_dir: Optional[Path] = None):
    """
    Load trained model from disk.

    Args:
        model_name: Name of the saved model file
        models_dir: Optional custom models directory

    Returns:
        Loaded model
    """
    models_path = models_dir or MODELS_DIR

    if not model_name.endswith('.joblib'):
        model_name = f"{model_name}.joblib"

    filepath = models_path / model_name
    model = joblib.load(filepath)
    logger.info(f"Loaded model from {filepath}")

    return model


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"):
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    logger.info(f"\n{model_name} Classification Report:")
    logger.info(f"\n{classification_report(y_true, y_pred)}")

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"\n{cm}")
