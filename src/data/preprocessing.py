"""
Data preprocessing module for Car Price Prediction.
Handles loading, cleaning, and splitting of raw car data.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from src.config import get_config


config = get_config()


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw car price data from CSV.

    Args:
        filepath: Path to CSV file. If None, uses config default.

    Returns:
        Raw DataFrame with all columns.
    """
    if filepath is None:
        raw_dir = Path(config.get("data.raw_dir", "data/raw"))
        filename = config.get("data.raw_filename", "car_prices.csv")
        filepath = raw_dir / filename

    df = pd.read_csv(filepath)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw car data.

    Operations:
    - Standardize column names
    - Handle missing values
    - Extract brand from car name
    - Parse numeric columns with units (mileage, engine, max_power)
    - Clean torque column (complex, dropped by default)
    - Remove duplicates
    - Handle outliers

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Extract brand from name (first word)
    if "name" in df.columns:
        df["brand"] = df["name"].apply(lambda x: str(x).split()[0] if pd.notna(x) else "unknown")

    # Parse numeric columns with units
    numeric_with_units = {
        "mileage": "kmpl|km/kg",
        "engine": "cc",
        "max_power": "bhp",
    }

    for col, unit_pattern in numeric_with_units.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r"([\d.]+)")[0]
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean seats column
    if "seats" in df.columns:
        df["seats"] = pd.to_numeric(df["seats"], errors="coerce")

    # Clean year
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Clean km_driven
    if "km_driven" in df.columns:
        df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")

    # Drop torque column (too messy, not reliable)
    if "torque" in df.columns:
        df = df.drop(columns=["torque"])

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")

    # Handle missing values
    df = handle_missing_values(df)

    # Handle outliers
    df = handle_outliers(df)

    print(f"Cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values based on config threshold."""
    threshold = config.get("preprocessing.missing_threshold", 0.5)
    missing_ratios = df.isnull().mean()

    # Drop columns with too many missing values
    cols_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Impute remaining missing values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["object", "category"]:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Handle outliers in numerical columns using IQR or Z-score method."""
    method = config.get("preprocessing.outlier_method", "iqr")
    threshold = config.get("preprocessing.outlier_threshold", 1.5)

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    target_col = config.get("data.target_column", "selling_price")

    for col in numerical_cols:
        if col == target_col:
            continue  # Don't remove outliers from target

        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
        else:  # zscore
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            lower = df[col].mean() - threshold * df[col].std()
            upper = df[col].mean() + threshold * df[col].std()

        outliers = (df[col] < lower) | (df[col] > upper)
        if outliers.any():
            print(f"  {col}: capping {outliers.sum()} outliers")
            df.loc[df[col] < lower, col] = lower
            df.loc[df[col] > upper, col] = upper

    return df


def split_data(
    df: pd.DataFrame,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    stratify: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Args:
        df: Cleaned DataFrame
        test_size: Fraction for test set
        random_state: Random seed
        stratify: Column to stratify on (for classification)

    Returns:
        (train_df, test_df)
    """
    if test_size is None:
        test_size = config.get("preprocessing.test_size", 0.2)
    if random_state is None:
        random_state = config.get("preprocessing.random_state", 42)

    # For regression, we can stratify by price bins
    if stratify is None and config.get("preprocessing.stratify_by_target", False):
        target_col = config.get("data.target_column", "selling_price")
        stratify = pd.qcut(df[target_col], q=10, duplicates="drop")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    print(f"Train: {train_df.shape[0]} rows, Test: {test_df.shape[0]} rows")
    return train_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> None:
    """Save processed train/test data as parquet."""
    if train_path is None:
        processed_dir = Path(config.get("data.processed_dir", "data/processed"))
        train_path = processed_dir / config.get("data.train_filename", "train.parquet")
    if test_path is None:
        processed_dir = Path(config.get("data.processed_dir", "data/processed"))
        test_path = processed_dir / config.get("data.test_filename", "test.parquet")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"Saved processed data to {train_path.parent}")


def load_processed_data(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train/test data from parquet."""
    if train_path is None:
        processed_dir = Path(config.get("data.processed_dir", "data/processed"))
        train_path = processed_dir / config.get("data.train_filename", "train.parquet")
    if test_path is None:
        processed_dir = Path(config.get("data.processed_dir", "data/processed"))
        test_path = processed_dir / config.get("data.test_filename", "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    print(f"Loaded processed - Train: {train_df.shape[0]}, Test: {test_df.shape[0]}")
    return train_df, test_df


def run_preprocessing_pipeline(raw_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing pipeline.

    Args:
        raw_path: Optional path to raw CSV

    Returns:
        (train_df, test_df) - processed and split data
    """
    print("=" * 50)
    print("Running Preprocessing Pipeline")
    print("=" * 50)

    # Load and clean
    df = load_raw_data(raw_path)
    df = clean_data(df)

    # Split
    train_df, test_df = split_data(df)

    # Save
    save_processed_data(train_df, test_df)

    print("=" * 50)
    print("Preprocessing Complete")
    print("=" * 50)

    return train_df, test_df


if __name__ == "__main__":
    run_preprocessing_pipeline()