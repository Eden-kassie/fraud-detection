"""Geolocation feature engineering for fraud detection."""
import pandas as pd
import numpy as np
from typing import Optional
from src.utils.logger import logger


def ip_to_integer(ip_address: str) -> int:
    """
    Convert IP address string to integer format.

    Args:
        ip_address: IP address string (e.g., '192.168.1.1')

    Returns:
        Integer representation of IP address
    """
    try:
        parts = ip_address.split('.')
        return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    except:
        return 0


def merge_ip_country(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    ip_col: str = 'ip_address'
) -> pd.DataFrame:
    """
    Merge fraud data with country information using IP address range lookup.

    Args:
        fraud_df: Fraud dataset with IP addresses
        ip_country_df: IP to country mapping dataset
        ip_col: Name of IP address column in fraud_df

    Returns:
        DataFrame with country information merged
    """
    logger.info("Merging IP addresses with country data...")

    df_merged = fraud_df.copy()

    # Convert IP addresses to integers
    logger.info("Converting IP addresses to integer format...")
    df_merged['ip_int'] = df_merged[ip_col].apply(ip_to_integer)

    # Ensure IP country data has integer ranges
    ip_ranges = ip_country_df.copy()
    if 'lower_bound_ip_address' in ip_ranges.columns:
        ip_ranges['lower_bound_ip_address'] = ip_ranges['lower_bound_ip_address'].astype('int64')
    if 'upper_bound_ip_address' in ip_ranges.columns:
        ip_ranges['upper_bound_ip_address'] = ip_ranges['upper_bound_ip_address'].astype('int64')

    # Convert ip_int in df_merged as well just to be certain
    df_merged['ip_int'] = df_merged['ip_int'].astype('int64')

    # Perform range-based merge using merge_asof
    logger.info("Performing range-based IP lookup...")

    # Sort both dataframes
    df_merged = df_merged.sort_values('ip_int')
    ip_ranges = ip_ranges.sort_values('lower_bound_ip_address')

    # Merge using merge_asof for range lookup
    df_merged = pd.merge_asof(
        df_merged,
        ip_ranges,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    # Filter to only keep matches within the range
    if 'upper_bound_ip_address' in df_merged.columns:
        df_merged = df_merged[
            (df_merged['ip_int'] >= df_merged['lower_bound_ip_address']) &
            (df_merged['ip_int'] <= df_merged['upper_bound_ip_address'])
        ]

    # Count successful matches
    if 'country' in df_merged.columns:
        matched = df_merged['country'].notna().sum()
        logger.info(f"Successfully matched {matched}/{len(fraud_df)} IP addresses to countries")

    return df_merged


def analyze_fraud_by_country(df: pd.DataFrame, country_col: str = 'country', fraud_col: str = 'class') -> pd.DataFrame:
    """
    Analyze fraud patterns by country.

    Args:
        df: DataFrame with country and fraud labels
        country_col: Name of country column
        fraud_col: Name of fraud label column

    Returns:
        DataFrame with fraud statistics by country
    """
    logger.info("Analyzing fraud patterns by country...")

    # Check if columns exist
    if country_col not in df.columns:
        error_msg = f"Country column '{country_col}' not found in DataFrame. Available columns: {df.columns.tolist()}"
        logger.error(error_msg)
        print(f"DEBUG ERROR: {error_msg}")
        return pd.DataFrame()

    actual_fraud_col = fraud_col
    if actual_fraud_col not in df.columns:
        # Check for case-insensitive match
        for c in df.columns:
            if c.lower() == fraud_col.lower():
                actual_fraud_col = c
                break

    if actual_fraud_col not in df.columns:
        error_msg = f"Fraud column '{fraud_col}' not found in DataFrame. Available columns: {df.columns.tolist()}"
        logger.error(error_msg)
        print(f"DEBUG ERROR: {error_msg}")
        return pd.DataFrame()

    country_stats = df.groupby(country_col).agg({
        actual_fraud_col: ['count', 'sum', 'mean']
    }).reset_index()

    country_stats.columns = [country_col, 'total_transactions', 'fraud_count', 'fraud_rate']
    country_stats = country_stats.sort_values('fraud_rate', ascending=False)

    logger.info(f"\nTop 10 countries by fraud rate:")
    logger.info(f"\n{country_stats.head(10)}")

    return country_stats


def create_country_features(
    df: pd.DataFrame,
    country_col: str = 'country',
    fraud_col: str = 'class'
) -> pd.DataFrame:
    """
    Create country-based risk features.

    Args:
        df: DataFrame with country information
        country_col: Name of country column
        fraud_col: Name of fraud label column

    Returns:
        DataFrame with country-based features
    """
    logger.info("Creating country-based risk features...")

    df_feat = df.copy()

    # Calculate country fraud rate
    country_fraud_rate = df_feat.groupby(country_col)[fraud_col].mean().to_dict()
    df_feat['country_fraud_rate'] = df_feat[country_col].map(country_fraud_rate)

    # Calculate country transaction count
    country_txn_count = df_feat.groupby(country_col).size().to_dict()
    df_feat['country_txn_count'] = df_feat[country_col].map(country_txn_count)

    # Country risk category based on fraud rate
    df_feat['country_risk'] = pd.cut(
        df_feat['country_fraud_rate'],
        bins=[0, 0.05, 0.1, 0.2, 1.0],
        labels=['low', 'medium', 'high', 'very_high'],
        include_lowest=True
    )

    logger.info("Created country-based risk features")

    return df_feat
