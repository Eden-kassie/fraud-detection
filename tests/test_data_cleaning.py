"""Tests for data cleaning functions."""
import pytest
import pandas as pd
import numpy as np
from src.data.cleaning import (
    handle_missing_values,
    remove_duplicates,
    correct_data_types,
    validate_data
)


def test_handle_missing_values_drop():
    """Test missing value handling with drop strategy."""
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [5, 6, 7, 8],
        'c': [np.nan, np.nan, np.nan, np.nan]
    })

    result = handle_missing_values(df, strategy='drop', threshold=0.5)

    # Column 'c' should be dropped (100% missing)
    assert 'c' not in result.columns
    # Rows with missing values should be dropped
    assert len(result) == 3


def test_handle_missing_values_mean():
    """Test missing value handling with mean imputation."""
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [5, 6, 7, 8]
    })

    result = handle_missing_values(df, strategy='mean')

    # No missing values should remain
    assert result.isnull().sum().sum() == 0
    # Mean of [1, 2, 4] is 2.33...
    assert result['a'].iloc[2] == pytest.approx(2.33, rel=0.1)


def test_remove_duplicates():
    """Test duplicate removal."""
    df = pd.DataFrame({
        'a': [1, 2, 2, 3],
        'b': [4, 5, 5, 6]
    })

    result = remove_duplicates(df)

    assert len(result) == 3


def test_correct_data_types():
    """Test data type correction."""
    df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'value': ['1', '2']
    })

    type_mapping = {
        'date': 'datetime',
        'value': 'int'
    }

    result = correct_data_types(df, type_mapping)

    assert pd.api.types.is_datetime64_any_dtype(result['date'])
    assert pd.api.types.is_integer_dtype(result['value'])


def test_validate_data():
    """Test data validation."""
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })

    # Should pass basic validation
    assert validate_data(df) == True

    # Empty dataframe should fail
    empty_df = pd.DataFrame()
    assert validate_data(empty_df) == False
