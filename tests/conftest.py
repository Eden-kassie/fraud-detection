"""Pytest configuration and fixtures."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_fraud_data():
    """Create sample fraud dataset for testing."""
    np.random.seed(42)
    n_samples = 1000

    data = {
        'user_id': np.arange(n_samples),
        'signup_time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'purchase_time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'purchase_value': np.random.uniform(10, 500, n_samples),
        'device_id': np.random.choice(['device_' + str(i) for i in range(100)], n_samples),
        'source': np.random.choice(['SEO', 'Ads', 'Direct'], n_samples),
        'browser': np.random.choice(['Chrome', 'Firefox', 'Safari'], n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'ip_address': [f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}" for _ in range(n_samples)],
        'class': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_creditcard_data():
    """Create sample credit card dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 28

    data = {
        'Time': np.arange(n_samples),
        'Amount': np.random.uniform(0, 1000, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }

    # Add V1-V28 features (PCA components)
    for i in range(1, n_features + 1):
        data[f'V{i}'] = np.random.randn(n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def sample_features_target(sample_fraud_data):
    """Create sample X, y split."""
    X = sample_fraud_data.drop(columns=['class'])
    y = sample_fraud_data['class']
    return X, y


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
