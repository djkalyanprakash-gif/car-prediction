"""
Tests for data preprocessing module.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import clean_data, split_data, handle_missing_values, handle_outliers


class TestCleanData:
    """Tests for clean_data function."""

    def test_clean_data_basic(self):
        """Test basic cleaning operations."""
        # Create sample raw data
        df = pd.DataFrame({
            "Name": ["Maruti Swift", "Hyundai i20"],
            "Year": [2018, 2019],
            "Kilometers_Driven": [50000, 30000],
            "Fuel_Type": ["Petrol", "Diesel"],
            "Transmission": ["Manual", "Automatic"],
            "Owner_Type": ["First", "Second"],
            "Mileage": ["20 kmpl", "25 kmpl"],
            "Engine": ["1200 CC", "1500 CC"],
            "Power": ["80 bhp", "100 bhp"],
            "Seats": [5, 5],
            "Price": [500000, 700000],
        })

        cleaned = clean_data(df)

        # Check column standardization
        assert "name" in cleaned.columns
        assert "year" in cleaned.columns
        assert "km_driven" in cleaned.columns
        assert "fuel" in cleaned.columns
        assert "transmission" in cleaned.columns
        assert "owner" in cleaned.columns
        assert "mileage" in cleaned.columns
        assert "engine" in cleaned.columns
        assert "max_power" in cleaned.columns
        assert "seats" in cleaned.columns
        assert "selling_price" in cleaned.columns

        # Check brand extraction
        assert "brand" in cleaned.columns
        assert cleaned["brand"].iloc[0] == "Maruti"
        assert cleaned["brand"].iloc[1] == "Hyundai"

        # Check numeric parsing
        assert cleaned["mileage"].iloc[0] == 20.0
        assert cleaned["engine"].iloc[0] == 1200.0
        assert cleaned["max_power"].iloc[0] == 80.0

        # Check torque dropped
        assert "torque" not in cleaned.columns

    def test_clean_data_missing_values(self):
        """Test missing value handling."""
        df = pd.DataFrame({
            "Name": ["Car A", "Car B", "Car C"],
            "Year": [2018, 2019, None],
            "Kilometers_Driven": [50000, None, 30000],
            "Fuel_Type": ["Petrol", "Diesel", "Petrol"],
            "Transmission": ["Manual", "Automatic", "Manual"],
            "Owner_Type": ["First", "Second", "First"],
            "Mileage": ["20 kmpl", "25 kmpl", "18 kmpl"],
            "Engine": ["1200 CC", "1500 CC", "1000 CC"],
            "Power": ["80 bhp", "100 bhp", "70 bhp"],
            "Seats": [5, 5, 5],
            "Price": [500000, 700000, 400000],
        })

        cleaned = clean_data(df)

        # No missing values should remain
        assert cleaned.isnull().sum().sum() == 0


class TestSplitData:
    """Tests for split_data function."""

    def test_split_data_shapes(self):
        """Test train/test split produces correct shapes."""
        df = pd.DataFrame({
            "year": range(2010, 2020),
            "km_driven": range(10000, 110000, 10000),
            "fuel": ["Petrol", "Diesel"] * 5,
            "transmission": ["Manual"] * 10,
            "owner": ["First"] * 10,
            "mileage": [20.0] * 10,
            "engine": [1200] * 10,
            "max_power": [80.0] * 10,
            "seats": [5] * 10,
            "brand": ["BrandA"] * 10,
            "selling_price": range(500000, 1500000, 100000),
        })

        train, test = split_data(df, test_size=0.2, random_state=42)

        assert len(train) == 8
        assert len(test) == 2
        assert len(train) + len(test) == len(df)

    def test_split_data_stratify(self):
        """Test stratification works."""
        df = pd.DataFrame({
            "year": [2015, 2015, 2015, 2015, 2020, 2020, 2020, 2020],
            "km_driven": [50000] * 8,
            "fuel": ["Petrol"] * 8,
            "transmission": ["Manual"] * 8,
            "owner": ["First"] * 8,
            "mileage": [20.0] * 8,
            "engine": [1200] * 8,
            "max_power": [80.0] * 8,
            "seats": [5] * 8,
            "brand": ["BrandA"] * 8,
            "selling_price": [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000],
        })

        train, test = split_data(df, test_size=0.25, random_state=42, stratify=df["selling_price"])

        # Should have roughly same distribution
        assert len(train) == 6
        assert len(test) == 2


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""

    def test_drop_high_missing_columns(self):
        """Test columns with high missing ratio are dropped."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, np.nan, np.nan, np.nan, np.nan],  # 80% missing
            "c": ["x", "y", "z", "x", "y"],
        })

        result = handle_missing_values(df)

        # Column b should be dropped (>50% missing)
        assert "b" not in result.columns
        assert "a" in result.columns
        assert "c" in result.columns

    def test_impute_remaining_missing(self):
        """Test remaining missing values are imputed."""
        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5],
            "b": ["x", "y", "z", np.nan, "x"],
        })

        result = handle_missing_values(df)

        assert result["a"].isnull().sum() == 0
        assert result["b"].isnull().sum() == 0


class TestHandleOutliers:
    """Tests for handle_outliers function."""

    def test_iqr_outlier_capping(self):
        """Test IQR method caps outliers."""
        df = pd.DataFrame({
            "normal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "with_outliers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],  # 1000 is outlier
        })

        result = handle_outliers(df)

        # Outlier should be capped
        assert result["with_outliers"].max() < 1000
        # Normal column unchanged
        assert result["normal"].max() == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])