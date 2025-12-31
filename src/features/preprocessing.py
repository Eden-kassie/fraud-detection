"""Preprocessing utilities for feature transformation."""
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils.logger import logger

def create_preprocessing_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    scale_strategy: str = 'standard',
    impute_strategy: str = 'median'
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline with scaling and encoding.

    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        scale_strategy: Scaling strategy ('standard', 'minmax', 'robust')
        impute_strategy: Imputation strategy for numerical missing values

    Returns:
        ColumnTransformer for preprocessing
    """
    logger.info(f"Creating preprocessing pipeline (Scaling: {scale_strategy}, Imputation: {impute_strategy})")

    # Numerical transformer
    if scale_strategy == 'standard':
        scaler = StandardScaler()
    elif scale_strategy == 'minmax':
        scaler = MinMaxScaler()
    elif scale_strategy == 'robust':
        scaler = RobustScaler()
    else:
        logger.warning(f"Unknown scale_strategy '{scale_strategy}', defaulting to StandardScaler")
        scaler = StandardScaler()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_strategy)),
        ('scaler', scaler)
    ])

    # Categorical transformer
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop columns not specified
    )

    return preprocessor
