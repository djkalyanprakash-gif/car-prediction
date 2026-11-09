"""
Feature engineering module for Car Price Prediction.
Creates meaningful features from cleaned data.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

from src.config import get_config


config = get_config()


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer.
    Fits on training data, transforms both train and test.
    """

    def __init__(self):
        self.categorical_columns = config.get("features.categorical_columns", [])
        self.numerical_columns = config.get("features.numerical_columns", [])
        self.target_encode_columns = config.get("features.target_encode_columns", [])
        self.drop_columns = config.get("features.drop_columns", [])
        self.create_age_feature = config.get("features.create_age_feature", True)
        self.create_km_per_year = config.get("features.create_km_per_year", True)

        # Fitted encoders
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder: Optional[TargetEncoder] = None
        self.target_mean_: Dict[str, float] = {}
        self.global_target_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Fit encoders on training data."""
        X = X.copy()

        # Create engineered features first
        X = self._create_features(X)

        # Fit label encoders for categorical columns
        for col in self.categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                # Handle unseen categories by adding 'unknown'
                values = X[col].fillna("unknown").astype(str).unique()
                le.fit(list(values) + ["unknown"])
                self.label_encoders[col] = le

        # Fit target encoder for high-cardinality columns
        if y is not None and self.target_encode_columns:
            self.global_target_mean_ = y.mean()
            cols_to_encode = [c for c in self.target_encode_columns if c in X.columns]
            if cols_to_encode:
                self.target_encoder = TargetEncoder(smooth="auto", target_type="continuous")
                self.target_encoder.fit(X[cols_to_encode], y)

                # Store target means for fallback
                for col in cols_to_encode:
                    self.target_mean_[col] = X.groupby(col)[y.name].mean().to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoders."""
        X = X.copy()

        # Create engineered features
        X = self._create_features(X)

        # Drop specified columns
        X = X.drop(columns=[c for c in self.drop_columns if c in X.columns], errors="ignore")

        # Label encode categorical columns
        for col, le in self.label_encoders.items():
            if col in X.columns:
                # Handle unseen categories
                values = X[col].fillna("unknown").astype(str)
                values = values.apply(lambda x: x if x in le.classes_ else "unknown")
                X[col] = le.transform(values)

        # Target encode high-cardinality columns
        if self.target_encoder is not None:
            cols_to_encode = [c for c in self.target_encode_columns if c in X.columns]
            if cols_to_encode:
                encoded = self.target_encoder.transform(X[cols_to_encode])
                # Rename to avoid collision
                encoded.columns = [f"{c}_te" for c in encoded.columns]
                X = X.drop(columns=cols_to_encode)
                X = pd.concat([X, encoded], axis=1)

        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        df = df.copy()

        # Car age (current year - manufacturing year)
        if self.create_age_feature and "year" in df.columns:
            current_year = pd.Timestamp.now().year
            df["car_age"] = current_year - df["year"]
            df["car_age"] = df["car_age"].clip(lower=0, upper=30)

        # Kilometers driven per year
        if self.create_km_per_year and "km_driven" in df.columns and "year" in df.columns:
            current_year = pd.Timestamp.now().year
            df["km_per_year"] = df["km_driven"] / (current_year - df["year"] + 1)
            df["km_per_year"] = df["km_per_year"].replace([np.inf, -np.inf], np.nan)
            df["km_per_year"] = df["km_per_year"].fillna(df["km_per_year"].median())

        # Power to weight ratio (approximate)
        # We don't have weight, but we can use engine size as proxy
        if "max_power" in df.columns and "engine" in df.columns:
            df["power_per_cc"] = df["max_power"] / (df["engine"] / 1000)
            df["power_per_cc"] = df["power_per_cc"].replace([np.inf, -np.inf], np.nan)
            df["power_per_cc"] = df["power_per_cc"].fillna(df["power_per_cc"].median())

        # Mileage categories
        if "mileage" in df.columns:
            df["mileage_category"] = pd.qcut(
                df["mileage"], q=4, labels=["low", "medium", "high", "very_high"], duplicates="drop"
            )

        # Age categories
        if "car_age" in df.columns:
            df["age_category"] = pd.cut(
                df["car_age"],
                bins=[-1, 3, 7, 12, 30],
                labels=["new", "recent", "used", "old"],
            )

        # Brand tier (based on median price - computed during fit)
        # This would need target info, so we skip here and use target encoding instead

        return df


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Build features for train and test sets.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_column: Name of target column

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if target_column is None:
        target_column = config.get("data.target_column", "selling_price")

    # Separate features and target
    y_train = train_df[target_column].copy()
    y_test = test_df[target_column].copy()

    X_train = train_df.drop(columns=[target_column])
    X_test = test_df.drop(columns=[target_column])

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Fit on train, transform both
    X_train = fe.fit_transform(X_train, y_train)
    X_test = fe.transform(X_test)

    # Align columns (in case test has missing categories)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    print(f"Feature engineering complete:")
    print(f"  Train features: {X_train.shape[1]}")
    print(f"  Test features: {X_test.shape[1]}")

    return X_train, X_test, y_train, y_test


def save_feature_engineer(fe: FeatureEngineer, path: Optional[Path] = None) -> None:
    """Save fitted feature engineer."""
    if path is None:
        path = Path("models/feature_engineer.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fe, path)
    print(f"Saved feature engineer to {path}")


def load_feature_engineer(path: Optional[Path] = None) -> FeatureEngineer:
    """Load fitted feature engineer."""
    if path is None:
        path = Path("models/feature_engineer.joblib")
    return joblib.load(path)


def get_feature_names(fe: FeatureEngineer) -> List[str]:
    """Get feature names after transformation (approximate)."""
    # This is approximate since we don't have transformed data here
    names = []
    for col in fe.numerical_columns:
        names.append(col)
    # Add engineered features
    if fe.create_age_feature:
        names.extend(["car_age", "age_category"])
    if fe.create_km_per_year:
        names.append("km_per_year")
    names.extend(["power_per_cc", "mileage_category"])
    # Encoded categoricals
    for col in fe.categorical_columns:
        if col in fe.target_encode_columns:
            names.append(f"{col}_te")
        else:
            names.append(col)
    return names


if __name__ == "__main__":
    # Quick test with dummy data
    from src.data.preprocessing import run_preprocessing_pipeline

    train_df, test_df = run_preprocessing_pipeline()
    X_train, X_test, y_train, y_test = build_features(train_df, test_df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")