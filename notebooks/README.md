# Notebooks Guide

This directory contains Jupyter notebooks for the fraud detection project. Execute them in the following order:

## 1. Exploratory Data Analysis (EDA)

### `eda-fraud-data.ipynb`
**Purpose**: Analyze the e-commerce fraud dataset

**Key Analyses**:
- Data overview and statistics
- Missing value analysis
- Class distribution (fraud vs non-fraud)
- Univariate analysis of features
- Bivariate analysis (features vs fraud)
- Correlation analysis
- Geolocation fraud patterns

**Prerequisites**: Raw data in `data/raw/`

---

### `eda-creditcard.ipynb`
**Purpose**: Analyze the credit card transaction dataset

**Key Analyses**:
- Data overview and statistics
- Class distribution
- PCA component distributions
- Transaction amount analysis
- Time-based patterns
- Correlation analysis

**Prerequisites**: Raw data in `data/raw/`

---

## 2. Feature Engineering

### `feature-engineering.ipynb`
**Purpose**: Create engineered features for both datasets

**Features Created**:
- **Time-based**: hour_of_day, day_of_week, time_since_signup
- **Transaction**: frequency, velocity, amount statistics
- **Geolocation**: IP-to-country mapping, country risk scores
- **Encoding**: categorical variables, scaling numerical features

**Output**: Processed data saved to `data/processed/`

**Prerequisites**: Completed EDA notebooks

---

## 3. Model Training

### `modeling.ipynb`
**Purpose**: Train and evaluate fraud detection models

**Models Trained**:
1. **Baseline**: Logistic Regression
2. **Ensemble Models**:
   - Random Forest
   - XGBoost
   - LightGBM

**Evaluation**:
- Stratified K-fold cross-validation
- Metrics: AUC-PR, F1-Score, Precision, Recall
- Confusion matrices
- Feature importance analysis
- Model comparison and selection

**Output**: Trained models saved to `models/`

**Prerequisites**: Completed feature engineering

---

## 4. Model Interpretation

### `shap-explainability.ipynb`
**Purpose**: Explain model predictions using SHAP

**Analyses**:
- SHAP summary plots
- Feature importance from SHAP values
- Individual prediction explanations
- Force plots for specific transactions
- Dependence plots

**Prerequisites**: Trained models in `models/`

---

## Execution Tips

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Kernel**: Use the Python environment where you installed the package

4. **Data**: Ensure raw data is in `data/raw/` before starting

5. **Order**: Follow the numbered sequence above for best results

6. **Reusability**: All notebooks use functions from the `src/` package for consistency

---

## Common Issues

**Import Errors**: Make sure you've installed the package in editable mode:
```bash
pip install -e .
```

**Data Not Found**: Check that data files are in `data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

**Memory Issues**: For large datasets, consider:
- Processing data in chunks
- Using a subset for initial exploration
- Increasing available RAM

---

## Output Files

After running all notebooks, you should have:
- `data/processed/fraud_featured.csv` - Engineered fraud dataset
- `data/processed/creditcard_featured.csv` - Engineered credit card dataset
- `models/*.joblib` - Trained model files
- Various plots and visualizations in the notebooks
