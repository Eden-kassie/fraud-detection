# Fraud Detection for E-commerce and Bank Transactions

A comprehensive machine learning system for detecting fraudulent transactions in e-commerce and credit card data. This project implements industry-standard practices for data analysis, feature engineering, model training, and evaluation.

## 🎯 Project Overview

This project analyzes two datasets:
- **E-commerce Fraud Data**: Transaction data with user behavior and geolocation features
- **Credit Card Transactions**: Anonymized credit card transaction data with PCA-transformed features

### Key Features

- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Advanced feature engineering (time-based, frequency, velocity features)
- ✅ Geolocation integration with IP-to-country mapping
- ✅ Class imbalance handling (SMOTE, undersampling)
- ✅ Multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- ✅ Stratified K-fold cross-validation
- ✅ Model interpretability with SHAP
- ✅ Comprehensive testing suite
- ✅ CI/CD with GitHub Actions

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Eden-kassie/Fraud-Detection.git
cd Fraud-Detection
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install package in editable mode
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### 4. Prepare Data

Place your datasets in the `data/raw/` directory:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

## 📁 Project Structure

```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD pipeline
├── .vscode/
│   └── settings.json              # VS Code configuration
├── data/
│   ├── raw/                       # Original datasets (gitignored)
│   └── processed/                 # Processed datasets (gitignored)
├── notebooks/
│   ├── eda-fraud-data.ipynb       # EDA for e-commerce data
│   ├── eda-creditcard.ipynb       # EDA for credit card data
│   ├── feature-engineering.ipynb  # Feature creation
│   ├── modeling.ipynb             # Model training & evaluation
│   ├── shap-explainability.ipynb  # Model interpretation
│   └── README.md                  # Notebook documentation
├── src/
│   ├── data/                      # Data loading and preprocessing
│   ├── features/                  # Feature engineering
│   ├── models/                    # Model training and evaluation
│   ├── visualization/             # Plotting utilities
│   └── utils/                     # Helper functions
├── tests/                         # Unit tests
├── scripts/                       # Utility scripts
├── models/                        # Saved model artifacts (gitignored)
├── requirements.txt               # Project dependencies
├── setup.py                       # Package configuration
├── .gitignore                     # Git ignore patterns
└── README.md                      # This file
```

## 📊 Usage

### Running Notebooks

Execute notebooks in the following order:

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook notebooks/eda-fraud-data.ipynb
   jupyter notebook notebooks/eda-creditcard.ipynb
   ```

2. **Feature Engineering**
   ```bash
   jupyter notebook notebooks/feature-engineering.ipynb
   ```

3. **Model Training**
   ```bash
   jupyter notebook notebooks/modeling.ipynb
   ```

4. **Model Interpretation**
   ```bash
   jupyter notebook notebooks/shap-explainability.ipynb
   ```

### Using the Python Package

```python
from src.data.loading import load_fraud_data, load_creditcard_data
from src.features.engineering import create_all_features
from src.models.baseline import LogisticRegressionBaseline
from src.models.ensemble import XGBoostModel

# Load data
fraud_df = load_fraud_data()

# Create features
fraud_df = create_all_features(fraud_df)

# Train model
model = XGBoostModel()
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(metrics)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_features.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

## 🎯 Model Performance

### E-commerce Fraud Detection
| Model | AUC-PR | F1-Score | Precision | Recall |
|-------|--------|----------|-----------|--------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD |

### Credit Card Fraud Detection
| Model | AUC-PR | F1-Score | Precision | Recall |
|-------|--------|----------|-----------|--------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD |

*Note: Metrics will be updated after model training*

## 🔬 Methodology

### Data Preprocessing
1. Handle missing values with appropriate imputation
2. Remove duplicate records
3. Correct data types
4. Merge geolocation data using IP address ranges

### Feature Engineering
- **Time-based features**: hour of day, day of week, time since signup
- **Transaction features**: frequency, velocity, amount statistics
- **Geolocation features**: country-based risk scores

### Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random undersampling
- Stratified sampling for train-test split

### Model Training
- Baseline: Logistic Regression
- Ensemble: Random Forest, XGBoost, LightGBM
- Hyperparameter tuning with cross-validation
- Stratified 5-fold cross-validation

### Evaluation Metrics
- **AUC-PR**: Area under Precision-Recall curve (primary metric for imbalanced data)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results
- **Cross-validation**: Mean and standard deviation across folds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Use Black for code formatting (line length: 100)
- Write unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Eden Moges - Initial work

## 🙏 Acknowledgments

- Dataset sources
- Scikit-learn, XGBoost, LightGBM communities
- SHAP for model interpretability

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

