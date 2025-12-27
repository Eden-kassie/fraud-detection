"""Configuration settings for the fraud detection project."""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
FRAUD_DATA_PATH = RAW_DATA_DIR / "Fraud_Data.csv"
IP_COUNTRY_PATH = RAW_DATA_DIR / "IpAddress_to_Country.csv"
CREDITCARD_PATH = RAW_DATA_DIR / "creditcard.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering configuration
TIME_WINDOWS = [1, 6, 12, 24]  # hours for transaction frequency features

# Model hyperparameters (default values)
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "eval_metric": "aucpr",
    "use_label_encoder": False,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "verbose": -1,
}
