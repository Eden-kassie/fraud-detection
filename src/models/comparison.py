"""Model comparison utilities."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.utils.logger import logger


def compare_models(
    model_results: List[Dict[str, any]],
    sort_by: str = 'auc_pr',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Compare multiple models based on their evaluation metrics.

    Args:
        model_results: List of dictionaries containing model results
                      Each dict should have 'model_name' and metric keys
        sort_by: Metric to sort by
        ascending: Sort order

    Returns:
        DataFrame with model comparison
    """
    logger.info("Comparing models...")

    comparison_df = pd.DataFrame(model_results)

    if sort_by in comparison_df.columns:
        comparison_df = comparison_df.sort_values(sort_by, ascending=ascending)

    logger.info(f"\nModel Comparison (sorted by {sort_by}):")
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    return comparison_df


def create_comparison_table(
    model_results: List[Dict[str, any]],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a formatted comparison table for models.

    Args:
        model_results: List of dictionaries containing model results
        metrics: Optional list of metrics to include

    Returns:
        Formatted comparison DataFrame
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'auc_pr']

    comparison_data = []

    for result in model_results:
        row = {'Model': result.get('model_name', 'Unknown')}
        for metric in metrics:
            if metric in result:
                row[metric.upper()] = f"{result[metric]:.4f}"
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    return comparison_df


def select_best_model(
    model_results: List[Dict[str, any]],
    primary_metric: str = 'auc_pr',
    secondary_metric: str = 'f1'
) -> Dict[str, any]:
    """
    Select the best model based on primary and secondary metrics.

    Args:
        model_results: List of dictionaries containing model results
        primary_metric: Primary metric for selection
        secondary_metric: Secondary metric (tiebreaker)

    Returns:
        Dictionary with best model information and justification
    """
    logger.info(f"Selecting best model based on {primary_metric}...")

    # Sort by primary metric
    sorted_results = sorted(
        model_results,
        key=lambda x: (x.get(primary_metric, 0), x.get(secondary_metric, 0)),
        reverse=True
    )

    best_model = sorted_results[0]

    justification = (
        f"Selected '{best_model['model_name']}' as the best model.\n"
        f"Justification:\n"
        f"  - Highest {primary_metric.upper()}: {best_model.get(primary_metric, 0):.4f}\n"
        f"  - {secondary_metric.upper()}: {best_model.get(secondary_metric, 0):.4f}\n"
    )

    # Add comparison with second best
    if len(sorted_results) > 1:
        second_best = sorted_results[1]
        improvement = (
            (best_model.get(primary_metric, 0) - second_best.get(primary_metric, 0))
            / second_best.get(primary_metric, 1e-10) * 100
        )
        justification += (
            f"  - Improvement over '{second_best['model_name']}': "
            f"{improvement:.2f}% in {primary_metric.upper()}\n"
        )

    logger.info(f"\n{justification}")

    return {
        'best_model': best_model,
        'justification': justification,
        'all_results': sorted_results
    }
