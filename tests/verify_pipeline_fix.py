import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.data.preprocessing import create_preprocessing_pipeline

def test_pipeline_creation():
    numeric_cols = ['purchase_value', 'age', 'time_since_signup', 'user_txn_count', 'user_avg_amount']
    categorical_cols = ['source', 'browser', 'sex', 'country_risk_level']

    try:
        pipeline = create_preprocessing_pipeline(
            numeric_cols,
            categorical_cols,
            scale_strategy='standard',
            encoding='onehot'
        )
        print("Preprocessing pipeline created successfully.")
        return True
    except TypeError as e:
        print(f"TypeError still occurring: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    if test_pipeline_creation():
        sys.exit(0)
    else:
        sys.exit(1)
