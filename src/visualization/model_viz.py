"""Visualization functions for model evaluation."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Union
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
    model_or_importances: Union[np.ndarray, object],
    feature_names: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance.

    Args:
        model_or_importances: Either a trained model with feature_importances_ attribute,
                             or an array of importance values
        feature_names: List of feature names
        top_n: Number of top features to display
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Extract importances from model if needed
    if hasattr(model_or_importances, 'feature_importances_'):
        importances = model_or_importances.feature_importances_
    else:
        importances = model_or_importances

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
    metrics: Optional[Union[str, List[str]]] = 'auc_pr',
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Compare multiple models side-by-side.

    Args:
        results_df: DataFrame with model results (columns: model_name, metrics)
        metrics: Single metric or list of metrics to compare
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib figure
    """
    # Handle single metric as string
    if isinstance(metrics, str):
        metrics = [metrics]

    # Auto-calculate figure size based on number of metrics
    if figsize is None:
        n_metrics = len(metrics)
        figsize = (12, 4 * n_metrics) if n_metrics > 1 else (12, 6)

    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)

    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in results_df.columns:
            results_sorted = results_df.sort_values(metric, ascending=False)
            ax.barh(results_sorted['model_name'], results_sorted[metric])
            ax.set_xlabel(metric.upper())
            ax.set_title(f'Model Comparison - {metric.upper()}')
            ax.invert_yaxis()

            # Add value labels
            for i, v in enumerate(results_sorted[metric]):
                ax.text(v, i, f' {v:.3f}', va='center')
        else:
            ax.text(0.5, 0.5, f"Metric '{metric}' not found",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Model Comparison - {metric.upper()} (Not Found)')

    plt.tight_layout()

    return fig


def plot_multi_roc_curve(
    results_list: List[Dict],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same axis.

    Args:
        results_list: List of dictionaries, each containing 'model' and 'model_name'
        X_test: Test features
        y_test: True labels
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for res in results_list:
        model = res['model']
        name = res['model_name']

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if isinstance(y_prob, np.ndarray) and len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
        elif hasattr(model, 'predict'):
            y_prob = model.predict(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multi-Model ROC Comparison')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_multi_pr_curve(
    results_list: List[Dict],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models on the same axis.

    Args:
        results_list: List of dictionaries, each containing 'model' and 'model_name'
        X_test: Test features
        y_test: True labels
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for res in results_list:
        model = res['model']
        name = res['model_name']

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if isinstance(y_prob, np.ndarray) and len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
        elif hasattr(model, 'predict'):
            y_prob = model.predict(X_test)
        else:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f'{name} (AUC-PR = {pr_auc:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Multi-Model Precision-Recall Comparison')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_class_distribution(
    y: Union[pd.Series, np.ndarray],
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot the distribution of classes.

    Args:
        y: Target variable values
        title: Title for the plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    counts = y.value_counts().sort_index()
    percentages = y.value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='viridis')

    # Add labels
    for i, count in enumerate(counts):
        pct = percentages.iloc[i]
        ax.text(i, count + (max(counts) * 0.01), f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(['Non-Fraud (0)', 'Fraud (1)'])

    plt.tight_layout()

    return fig
