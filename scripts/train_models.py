"""Script to train fraud detection models."""
import argparse
import pandas as pd
import numpy as np
import os
from typing import List, Dict
from src.utils.logger import logger
from src.utils.config import (
    FRAUD_DATA_PATH, CREDITCARD_PATH, MODELS_DIR,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS
)
from src.features.engineering import create_all_features
from src.features.preprocessing import create_preprocessing_pipeline
from src.models.data_prep import prepare_model_data, apply_class_imbalance_handling
from src.models.baseline import LogisticRegressionBaseline
from src.models.ensemble import RandomForestModel
from src.models.evaluation import save_model
from src.visualization.model_viz import plot_class_distribution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--dataset", type=str, default="fraud", choices=["fraud", "creditcard"],
                        help="Dataset to use ('fraud' or 'creditcard')")
    parser.add_argument("--output", type=str, default=str(MODELS_DIR),
                        help="Directory to save trained models")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS,
                        help="Number of cross-validation folds")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help="Test set proportion")
    parser.add_argument("--apply-smote", action="store_true",
                        help="Apply SMOTE for class imbalance")
    parser.add_argument("--scale-strategy", type=str, default="standard",
                        choices=["standard", "minmax", "robust"],
                        help="Scaling strategy for numerical features")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting training with args: {args}")

    # 1. Load Data
    data_path = FRAUD_DATA_PATH if args.dataset == "fraud" else CREDITCARD_PATH
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Feature Engineering
    if args.dataset == "fraud":
        # For fraud data, we apply our custom engineering
        df = create_all_features(df)
        target_col = 'class'
        # Define columns for preprocessing
        # Identify numeric and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)

        # Categorical columns (simple heuristic)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Remove high-cardinality or irrelevant columns
        exclude_cats = ['user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address']
        categorical_cols = [c for c in categorical_cols if c not in exclude_cats]
    else:
        # Credit card data is already engineered
        target_col = 'Class'
        numerical_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        categorical_cols = []

    # 3. Data Preparation (Split)
    X_train, X_test, y_train, y_test = prepare_model_data(
        df, target_col=target_col, test_size=args.test_size, random_state=RANDOM_STATE
    )

    # 4. Preprocessing Pipeline
    # We first fit the preprocessor on the original training data
    preprocessor = create_preprocessing_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        scale_strategy=args.scale_strategy
    )

    logger.info("Fitting preprocessor and transforming data...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # 5. Class Imbalance Handling (Only on training set)
    if args.apply_smote:
        # Before resampling, plot original distribution
        fig_before = plot_class_distribution(y_train, title=f"Class Distribution Before Resampling ({args.dataset})")
        fig_before.savefig(os.path.join(args.output, f"class_dist_before_{args.dataset}.png"))
        plt.close(fig_before)

        # Apply SMOTE to preprocessed data
        X_train_resampled, y_train_resampled = apply_class_imbalance_handling(
            X_train_preprocessed, y_train, method='smote'
        )

        # After resampling, plot new distribution
        fig_after = plot_class_distribution(y_train_resampled, title=f"Class Distribution After SMOTE ({args.dataset})")
        fig_after.savefig(os.path.join(args.output, f"class_dist_after_{args.dataset}.png"))
        plt.close(fig_after)
    else:
        logger.info("SMOTE not applied.")
        X_train_resampled, y_train_resampled = X_train_preprocessed, y_train

    # 6. Model Training
    models = {
        "baseline": LogisticRegressionBaseline(),
        "random_forest": RandomForestModel()
    }

    results = {}
    for name, model_wrapper in models.items():
        logger.info(f"Training {name} model...")

        # Train on (potentially resampled) preprocessed data
        model_wrapper.train(X_train_resampled, y_train_resampled)

        # Evaluate on preprocessed test data
        metrics = model_wrapper.evaluate(X_test_preprocessed, y_test)

        results[name] = metrics

        # Save model
        save_model(model_wrapper.model, f"{name}_{args.dataset}", args.output)

    logger.info("Training script completed successfully.")

if __name__ == "__main__":
    main()
