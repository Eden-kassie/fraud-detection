# Scripts

This directory contains utility scripts for training and evaluating fraud detection models.

## Available Scripts

### `train_models.py`
Train all fraud detection models on a specified dataset.

**Usage**:
```bash
python scripts/train_models.py --dataset fraud --output models/
```

**Arguments**:
- `--dataset`: Dataset to use ('fraud' or 'creditcard')
- `--output`: Directory to save trained models
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--test-size`: Test set proportion (default: 0.2)
- `--apply-smote`: Apply SMOTE for class imbalance

**Example**:
```bash
# Train on fraud data with SMOTE
python scripts/train_models.py --dataset fraud --apply-smote

# Train on credit card data with custom CV folds
python scripts/train_models.py --dataset creditcard --cv-folds 10
```

---

### `evaluate_models.py`
Evaluate saved models and generate comparison reports.

**Usage**:
```bash
python scripts/evaluate_models.py --models-dir models/ --dataset fraud
```

**Arguments**:
- `--models-dir`: Directory containing saved models
- `--dataset`: Dataset to evaluate on ('fraud' or 'creditcard')
- `--output`: Output file for evaluation report

**Example**:
```bash
# Evaluate all models
python scripts/evaluate_models.py --models-dir models/ --dataset fraud --output results.csv
```

---

## Development

To create new scripts:

1. Import necessary modules from `src/`
2. Use argparse for command-line arguments
3. Add logging for progress tracking
4. Follow the existing script structure
5. Update this README with usage instructions

---

## Notes

- Scripts use the same configuration as notebooks (from `src/utils/config.py`)
- All scripts assume the package is installed: `pip install -e .`
- Logs are printed to console by default
- Use `--help` flag with any script to see all options
