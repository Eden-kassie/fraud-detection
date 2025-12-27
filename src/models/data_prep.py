"""Data preparation utilities for model training."""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.config import TEST_SIZE, RANDOM_STATE


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split to preserve class distribution.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of test set
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Performing stratified train-test split (test_size={test_size})...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Log class distribution
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    logger.info(f"Train class distribution:\n{train_dist}")
    logger.info(f"Test class distribution:\n{test_dist}")

    return X_train, X_test, y_train, y_test


def separate_features_target(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features from target variable.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        exclude_cols: Optional list of columns to exclude from features

    Returns:
        Tuple of (X, y)
    """
    logger.info(f"Separating features from target '{target_col}'...")

    # Get target
    y = df[target_col]

    # Get features
    cols_to_exclude = [target_col]
    if exclude_cols:
        cols_to_exclude.extend(exclude_cols)

    X = df.drop(columns=cols_to_exclude, errors='ignore')

    logger.info(f"Features: {len(X.columns)} columns")
    logger.info(f"Target: {target_col}")
    logger.info(f"Total samples: {len(X)}")

    return X, y


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Optional[List[str]] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete data preparation pipeline for model training.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        exclude_cols: Optional list of columns to exclude from features
        test_size: Proportion of test set
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Starting complete data preparation pipeline...")

    # Separate features and target
    X, y = separate_features_target(df, target_col, exclude_cols)

    # Train-test split
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size, random_state
    )

    logger.info("Data preparation complete")

    return X_train, X_test, y_train, y_test
