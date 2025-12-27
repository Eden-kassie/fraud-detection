"""Feature engineering functions for fraud detection."""
import pandas as pd
import numpy as np
from typing import List, Optional
from src.utils.logger import logger
from src.utils.config import TIME_WINDOWS


def create_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Create time-based features from a datetime column.

    Args:
        df: Input DataFrame
        datetime_col: Name of the datetime column

    Returns:
        DataFrame with additional time features
    """
    df_feat = df.copy()

    # Ensure datetime column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_feat[datetime_col]):
        df_feat[datetime_col] = pd.to_datetime(df_feat[datetime_col])

    # Extract time components
    df_feat['hour_of_day'] = df_feat[datetime_col].dt.hour
    df_feat['day_of_week'] = df_feat[datetime_col].dt.dayofweek
    df_feat['day_of_month'] = df_feat[datetime_col].dt.day
    df_feat['month'] = df_feat[datetime_col].dt.month
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)

    # Time of day categories
    df_feat['time_of_day'] = pd.cut(
        df_feat['hour_of_day'],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )

    logger.info(f"Created time features from {datetime_col}")

    return df_feat


def create_time_since_signup(
    df: pd.DataFrame,
    signup_col: str,
    purchase_col: str,
    unit: str = 'hours'
) -> pd.DataFrame:
    """
    Calculate time elapsed between signup and purchase.

    Args:
        df: Input DataFrame
        signup_col: Name of signup datetime column
        purchase_col: Name of purchase datetime column
        unit: Time unit ('hours', 'days', 'minutes')

    Returns:
        DataFrame with time_since_signup feature
    """
    df_feat = df.copy()

    # Ensure datetime columns
    for col in [signup_col, purchase_col]:
        if not pd.api.types.is_datetime64_any_dtype(df_feat[col]):
            df_feat[col] = pd.to_datetime(df_feat[col])

    # Calculate time difference
    time_diff = df_feat[purchase_col] - df_feat[signup_col]

    if unit == 'hours':
        df_feat['time_since_signup'] = time_diff.dt.total_seconds() / 3600
    elif unit == 'days':
        df_feat['time_since_signup'] = time_diff.dt.total_seconds() / 86400
    elif unit == 'minutes':
        df_feat['time_since_signup'] = time_diff.dt.total_seconds() / 60
    else:
        raise ValueError(f"Unknown unit: {unit}")

    logger.info(f"Created time_since_signup feature in {unit}")

    return df_feat


def create_transaction_frequency(
    df: pd.DataFrame,
    user_col: str,
    datetime_col: str,
    time_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create transaction frequency features for each user.

    Args:
        df: Input DataFrame
        user_col: Name of user identifier column
        datetime_col: Name of datetime column
        time_windows: List of time windows in hours (default from config)

    Returns:
        DataFrame with transaction frequency features
    """
    df_feat = df.copy()

    if time_windows is None:
        time_windows = TIME_WINDOWS

    # Ensure datetime column
    if not pd.api.types.is_datetime64_any_dtype(df_feat[datetime_col]):
        df_feat[datetime_col] = pd.to_datetime(df_feat[datetime_col])

    # Sort by user and datetime
    df_feat = df_feat.sort_values([user_col, datetime_col])

    for window_hours in time_windows:
        feature_name = f'txn_freq_{window_hours}h'

        # Count transactions within time window for each user
        df_feat[feature_name] = df_feat.groupby(user_col)[datetime_col].transform(
            lambda x: x.rolling(f'{window_hours}h', on=x).count()
        )

        logger.info(f"Created {feature_name} feature")

    return df_feat


def create_transaction_velocity(
    df: pd.DataFrame,
    user_col: str,
    datetime_col: str,
    amount_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Create transaction velocity features (rate of transactions).

    Args:
        df: Input DataFrame
        user_col: Name of user identifier column
        datetime_col: Name of datetime column
        amount_col: Optional name of transaction amount column

    Returns:
        DataFrame with velocity features
    """
    df_feat = df.copy()

    # Ensure datetime column
    if not pd.api.types.is_datetime64_any_dtype(df_feat[datetime_col]):
        df_feat[datetime_col] = pd.to_datetime(df_feat[datetime_col])

    # Sort by user and datetime
    df_feat = df_feat.sort_values([user_col, datetime_col])

    # Time since last transaction for same user
    df_feat['time_since_last_txn'] = df_feat.groupby(user_col)[datetime_col].diff().dt.total_seconds() / 3600

    # Transaction count per user
    df_feat['user_txn_count'] = df_feat.groupby(user_col).cumcount() + 1

    # Average time between transactions
    df_feat['avg_time_between_txn'] = df_feat.groupby(user_col)['time_since_last_txn'].transform('mean')

    if amount_col and amount_col in df_feat.columns:
        # Amount-based velocity features
        df_feat['user_total_amount'] = df_feat.groupby(user_col)[amount_col].cumsum()
        df_feat['user_avg_amount'] = df_feat.groupby(user_col)[amount_col].transform('mean')
        df_feat['amount_vs_avg'] = df_feat[amount_col] / (df_feat['user_avg_amount'] + 1e-6)

        logger.info("Created transaction velocity features with amount")
    else:
        logger.info("Created transaction velocity features without amount")

    return df_feat


def create_all_features(
    df: pd.DataFrame,
    datetime_col: str = 'purchase_time',
    signup_col: Optional[str] = 'signup_time',
    user_col: Optional[str] = 'user_id',
    amount_col: Optional[str] = 'purchase_value'
) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Args:
        df: Input DataFrame
        datetime_col: Name of purchase datetime column
        signup_col: Name of signup datetime column (optional)
        user_col: Name of user identifier column (optional)
        amount_col: Name of amount column (optional)

    Returns:
        DataFrame with all engineered features
    """
    logger.info("Starting comprehensive feature engineering...")
    df_feat = df.copy()

    # Time-based features
    if datetime_col in df_feat.columns:
        df_feat = create_time_features(df_feat, datetime_col)

    # Time since signup
    if signup_col and signup_col in df_feat.columns and datetime_col in df_feat.columns:
        df_feat = create_time_since_signup(df_feat, signup_col, datetime_col)

    # Transaction frequency and velocity
    if user_col and user_col in df_feat.columns and datetime_col in df_feat.columns:
        df_feat = create_transaction_velocity(df_feat, user_col, datetime_col, amount_col)
        # Note: transaction_frequency requires rolling windows which may be slow on large datasets
        # Uncomment if needed:
        # df_feat = create_transaction_frequency(df_feat, user_col, datetime_col)

    logger.info(f"Feature engineering complete. Total features: {len(df_feat.columns)}")

    return df_feat
