"""Tests for feature engineering functions."""
import pytest
import pandas as pd
import numpy as np
from src.features.engineering import (
    create_time_features,
    create_time_since_signup,
    create_transaction_velocity
)


def test_create_time_features():
    """Test time feature creation."""
    df = pd.DataFrame({
        'purchase_time': pd.date_range('2023-01-01', periods=100, freq='H')
    })

    result = create_time_features(df, 'purchase_time')

    assert 'hour_of_day' in result.columns
    assert 'day_of_week' in result.columns
    assert 'is_weekend' in result.columns
    assert result['hour_of_day'].min() >= 0
    assert result['hour_of_day'].max() <= 23


def test_create_time_since_signup():
    """Test time since signup calculation."""
    df = pd.DataFrame({
        'signup_time': pd.date_range('2023-01-01', periods=10, freq='D'),
        'purchase_time': pd.date_range('2023-01-02', periods=10, freq='D')
    })

    result = create_time_since_signup(df, 'signup_time', 'purchase_time', unit='hours')

    assert 'time_since_signup' in result.columns
    # Should be 24 hours for all rows
    assert all(result['time_since_signup'] == 24.0)


def test_create_transaction_velocity():
    """Test transaction velocity feature creation."""
    df = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2],
        'purchase_time': pd.date_range('2023-01-01', periods=5, freq='H'),
        'purchase_value': [100, 200, 150, 300, 250]
    })

    result = create_transaction_velocity(df, 'user_id', 'purchase_time', 'purchase_value')

    assert 'user_txn_count' in result.columns
    assert 'user_total_amount' in result.columns
    assert 'user_avg_amount' in result.columns

    # User 1 should have 3 transactions
    user1_data = result[result['user_id'] == 1]
    assert user1_data['user_txn_count'].max() == 3
