"""Visualization functions for model evaluation."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from src.utils.logger import logger

sns.set_style("whitegrid")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels or ['Non-Fraud', 'Fraud'],
                yticklabels=labels or ['Non-Fraud', 'Fraud'])
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve with AUC-PR score.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color='darkorange', lw=2, label=f'{model_name} (AUC-PR = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to display
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Create DataFrame and sort
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(feat_imp_df['feature'], feat_imp_df['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.invert_yaxis()
    plt.tight_layout()

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'auc_pr',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Compare multiple models side-by-side.

    Args:
        results_df: DataFrame with model results (columns: model_name, metrics)
        metric: Metric to compare
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if metric in results_df.columns:
        results_sorted = results_df.sort_values(metric, ascending=False)
        ax.barh(results_sorted['model_name'], results_sorted[metric])
        ax.set_xlabel(metric.upper())
        ax.set_title(f'Model Comparison - {metric.upper()}')
        ax.invert_yaxis()

        # Add value labels
        for i, v in enumerate(results_sorted[metric]):
            ax.text(v, i, f' {v:.3f}', va='center')

    plt.tight_layout()

    return fig
