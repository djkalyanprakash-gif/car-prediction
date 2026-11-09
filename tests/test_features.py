"""
Tests for feature engineering module.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.features.build_features import FeatureEngineer, build_features


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def create_sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "year": np.random.randint(2010, 2023, n),
            "km_driven": np.random.randint(10000, 150000, n),
            "mileage": np.random.uniform(15, 30, n),
            "engine": np.random.choice([1000, 1200, 1500, 2000], n),
            "max_power": np.random.uniform(60, 150, n),
            "seats": np.random.choice([5, 7], n),
            "fuel": np.random.choice(["Petrol", "Diesel", "CNG"], n),
            "seller_type": np.random.choice(["Individual", "Dealer"], n),
            "transmission": np.random.choice(["Manual", "Automatic"], n),
            "owner": np.random.choice(["First Owner", "Second Owner", "Third Owner"], n),
            "brand": np.random.choice(["Maruti", "Hyundai", "Honda", "Toyota", "Mahindra"], n),
        })

    def test_fit_transform_shapes(self):
        """Test fit_transform preserves row count."""
        X = self.create_sample_data()
        y = pd.Series(np.random.uniform(300000, 1500000, len(X)), name="selling_price")

        fe = FeatureEngineer()
        X_transformed = fe.fit_transform(X, y)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] >= X.shape[1]  # May have more features after encoding

    def test_engineered_features_created(self):
        """Test that engineered features are created."""
        X = self.create_sample_data()
        y = pd.Series(np.random.uniform(300000, 1500000, len(X)), name="selling_price")

        fe = FeatureEngineer()
        X_transformed = fe.fit_transform(X, y)

        # Check engineered features exist
        assert "car_age" in X_transformed.columns
        assert "km_per_year" in X_transformed.columns
        assert "power_per_cc" in X_transformed.columns
        assert "mileage_category" in X_transformed.columns
        assert "age_category" in X_transformed.columns

    def test_categorical_encoding(self):
        """Test categorical columns are encoded."""
        X = self.create_sample_data()
        y = pd.Series(np.random.uniform(300000, 1500000, len(X)), name="selling_price")

        fe = FeatureEngineer()
        X_transformed = fe.fit_transform(X, y)

        # Original categorical columns should be encoded (numeric)
        for col in ["fuel", "seller_type", "transmission", "owner"]:
            assert X_transformed[col].dtype in ["int64", "int32", "float64"]

        # Brand should be target encoded
        assert "brand_te" in X_transformed.columns

    def test_transform_only(self):
        """Test transform on new data using fitted encoder."""
        X_train = self.create_sample_data()
        y_train = pd.Series(np.random.uniform(300000, 1500000, len(X_train)), name="selling_price")

        X_test = self.create_sample_data().iloc[:10]  # Smaller test set
        y_test = pd.Series(np.random.uniform(300000, 1500000, len(X_test)), name="selling_price")

        fe = FeatureEngineer()
        fe.fit(X_train, y_train)

        X_test_transformed = fe.transform(X_test)

        assert X_test_transformed.shape[0] == X_test.shape[0]
        assert list(X_test_transformed.columns) == list(fe.transform(X_train).columns)

    def test_unseen_category_handling(self):
        """Test unseen categories are handled gracefully."""
        X_train = self.create_sample_data()
        y_train = pd.Series(np.random.uniform(300000, 1500000, len(X_train)), name="selling_price")

        # Test data with unseen brand
        X_test = X_train.iloc[:5].copy()
        X_test.loc[0, "brand"] = "UnseenBrand"

        fe = FeatureEngineer()
        fe.fit(X_train, y_train)

        # Should not raise error
        X_test_transformed = fe.transform(X_test)
        assert X_test_transformed.shape[0] == 5

    def test_dropped_columns(self):
        """Test specified columns are dropped."""
        X = self.create_sample_data()
        # Add a column that should be dropped
        X["name"] = "Test Car"
        X["torque"] = "100 Nm"

        y = pd.Series(np.random.uniform(300000, 1500000, len(X)), name="selling_price")

        fe = FeatureEngineer()
        X_transformed = fe.fit_transform(X, y)

        assert "name" not in X_transformed.columns
        assert "torque" not in X_transformed.columns


class TestBuildFeatures:
    """Tests for build_features function."""

    def test_build_features_output(self):
        """Test build_features returns correct structure."""
        # Create train/test with target
        train = pd.DataFrame({
            "year": [2018, 2019, 2020],
            "km_driven": [50000, 30000, 20000],
            "mileage": [20.0, 25.0, 18.0],
            "engine": [1200, 1500, 1000],
            "max_power": [80.0, 100.0, 70.0],
            "seats": [5, 5, 5],
            "fuel": ["Petrol", "Diesel", "Petrol"],
            "seller_type": ["Individual", "Dealer", "Individual"],
            "transmission": ["Manual", "Automatic", "Manual"],
            "owner": ["First Owner", "Second Owner", "First Owner"],
            "brand": ["Maruti", "Hyundai", "Honda"],
            "selling_price": [500000, 700000, 400000],
        })

        test = train.copy()
        test["selling_price"] = [600000, 800000, 500000]

        X_train, X_test, y_train, y_test = build_features(train, test)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) == 3
        assert len(X_test) == 3
        assert len(y_train) == 3
        assert len(y_test) == 3
        assert list(X_train.columns) == list(X_test.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])