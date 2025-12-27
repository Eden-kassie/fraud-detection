"""Models module exports."""
from src.models.data_prep import (
    stratified_train_test_split,
    separate_features_target,
    prepare_model_data,
)
from src.models.evaluation import (
    calculate_metrics,
    stratified_kfold_cv,
    aggregate_cv_results,
    save_model,
    load_model,
    print_classification_report,
)
from src.models.baseline import LogisticRegressionBaseline
from src.models.ensemble import (
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    tune_hyperparameters,
)
from src.models.comparison import (
    compare_models,
    create_comparison_table,
    select_best_model,
)

__all__ = [
    # Data preparation
    "stratified_train_test_split",
    "separate_features_target",
    "prepare_model_data",
    # Evaluation
    "calculate_metrics",
    "stratified_kfold_cv",
    "aggregate_cv_results",
    "save_model",
    "load_model",
    "print_classification_report",
    # Baseline
    "LogisticRegressionBaseline",
    # Ensemble
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "tune_hyperparameters",
    # Comparison
    "compare_models",
    "create_comparison_table",
    "select_best_model",
]
