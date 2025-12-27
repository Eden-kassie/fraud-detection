"""Visualization functions for exploratory data analysis."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from src.utils.logger import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_univariate(
    df: pd.DataFrame,
    column: str,
    plot_type: str = "auto",
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create univariate distribution plot for a single variable.

    Args:
        df: Input DataFrame
        column: Column name to plot
        plot_type: Type of plot ('hist', 'box', 'auto')
        bins: Number of bins for histogram
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "auto":
        # Determine plot type based on data type
        if pd.api.types.is_numeric_dtype(df[column]):
            plot_type = "hist"
        else:
            plot_type = "count"

    if plot_type == "hist":
        df[column].hist(bins=bins, ax=ax, edgecolor='black')
        ax.set_ylabel('Frequency')
    elif plot_type == "box":
        df.boxplot(column=column, ax=ax)
    elif plot_type == "count":
        df[column].value_counts().plot(kind='bar', ax=ax)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)

    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    plt.tight_layout()

    return fig


def plot_bivariate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    plot_type: str = "auto",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create bivariate plot showing relationship between two variables.

    Args:
        df: Input DataFrame
        x_col: X-axis column name
        y_col: Y-axis column name
        plot_type: Type of plot ('scatter', 'box', 'violin', 'auto')
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "auto":
        # Determine plot type based on data types
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            plot_type = "scatter"
        else:
            plot_type = "box"

    if plot_type == "scatter":
        ax.scatter(df[x_col], df[y_col], alpha=0.5)
    elif plot_type == "box":
        df.boxplot(column=y_col, by=x_col, ax=ax)
        plt.suptitle('')  # Remove default title
    elif plot_type == "violin":
        sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)

    ax.set_title(f'{y_col} vs {x_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()

    return fig


def plot_class_distribution(
    y: pd.Series,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Visualize class distribution with counts and percentages.

    Args:
        y: Target variable
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Count plot
    class_counts = y.value_counts()
    class_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
    ax1.set_title(f'{title} - Counts')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    # Add count labels on bars
    for i, v in enumerate(class_counts):
        ax1.text(i, v, f'{v:,}', ha='center', va='bottom')

    # Pie chart
    class_counts.plot(kind='pie', ax=ax2, autopct='%1.2f%%', startangle=90)
    ax2.set_title(f'{title} - Percentages')
    ax2.set_ylabel('')

    plt.tight_layout()

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = False
) -> plt.Figure:
    """
    Create correlation heatmap for numerical features.

    Args:
        df: Input DataFrame
        columns: Optional list of columns to include
        figsize: Figure size
        annot: Whether to annotate cells with correlation values

    Returns:
        Matplotlib figure
    """
    if columns:
        corr_df = df[columns]
    else:
        # Select only numeric columns
        corr_df = df.select_dtypes(include=[np.number])

    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()

    return fig


def plot_fraud_by_country(
    df: pd.DataFrame,
    country_col: str = 'country',
    fraud_col: str = 'class',
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Visualize fraud patterns by country.

    Args:
        df: DataFrame with country and fraud information
        country_col: Name of country column
        fraud_col: Name of fraud label column
        top_n: Number of top countries to display
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Calculate fraud statistics by country
    country_stats = df.groupby(country_col).agg({
        fraud_col: ['count', 'sum', 'mean']
    }).reset_index()
    country_stats.columns = [country_col, 'total_txn', 'fraud_count', 'fraud_rate']

    # Get top countries by transaction count
    top_countries = country_stats.nlargest(top_n, 'total_txn')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Transaction volume by country
    ax1.barh(top_countries[country_col], top_countries['total_txn'])
    ax1.set_xlabel('Total Transactions')
    ax1.set_title(f'Top {top_n} Countries by Transaction Volume')
    ax1.invert_yaxis()

    # Fraud rate by country
    fraud_rate_sorted = top_countries.sort_values('fraud_rate', ascending=False)
    colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green'
              for x in fraud_rate_sorted['fraud_rate']]
    ax2.barh(fraud_rate_sorted[country_col], fraud_rate_sorted['fraud_rate'], color=colors)
    ax2.set_xlabel('Fraud Rate')
    ax2.set_title(f'Fraud Rate by Country (Top {top_n})')
    ax2.invert_yaxis()

    plt.tight_layout()

    return fig
