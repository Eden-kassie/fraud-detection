"""Tests for model training and evaluation."""
import pytest
import pandas as pd
import numpy as np
from src.models.baseline import LogisticRegressionBaseline
from src.models.ensemble import RandomForestModel
from src.models.evaluation import calculate_metrics, save_model, load_model
from src.models.data_prep import stratified_train_test_split


def test_logistic_regression_training(sample_features_target):
    """Test Logistic Regression model training."""
    X, y = sample_features_target

    # Select only numeric columns for simplicity
    X_numeric = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = stratified_train_test_split(X_numeric, y, test_size=0.2)

    model = LogisticRegressionBaseline(max_iter=100)
    model.train(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})


def test_random_forest_training(sample_features_target):
    """Test Random Forest model training."""
    X, y = sample_features_target

    # Select only numeric columns
    X_numeric = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = stratified_train_test_split(X_numeric, y, test_size=0.2)

    model = RandomForestModel(n_estimators=10, max_depth=3)
    model.train(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})


def test_calculate_metrics():
    """Test metrics calculation."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 'auc_pr' in metrics

    # Check value ranges
    for metric_value in metrics.values():
        assert 0 <= metric_value <= 1


def test_model_save_load(sample_features_target, temp_model_dir):
    """Test model saving and loading."""
    X, y = sample_features_target
    X_numeric = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = stratified_train_test_split(X_numeric, y, test_size=0.2)

    model = LogisticRegressionBaseline(max_iter=100)
    model.train(X_train, y_train)

    # Save model
    save_model(model.model, "test_model", temp_model_dir)

    # Load model
    loaded_model = load_model("test_model", temp_model_dir)

    # Test predictions are the same
    original_pred = model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)

    assert np.array_equal(original_pred, loaded_pred)
